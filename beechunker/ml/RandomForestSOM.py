import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from beechunker.common.beechunker_logging import setup_logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from beechunker.common.config import config
from beechunker.ml.feature_extraction import OptimalThroughputProcessor

logger = setup_logging("rf_classifier")

class BeeChunkerRF:
    """
    Random Forest Classifier for binary throughput prediction and optimal chunk size selection.
    """
    def __init__(self):
        """Initialize the Random Forest classifier and environment."""
        self.model = None
        self.models_dir = config.get("ml", "models_dir")
        os.makedirs(self.models_dir, exist_ok=True)

    def set_last_training_time(self):
        path = os.path.join(self.models_dir, "last_training_time.txt")
        with open(path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_last_training_time(self):
        path = os.path.join(self.models_dir, "last_training_time.txt")
        try:
            with open(path, "r") as f:
                ts = f.read().strip()
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            logger.warning("Last training time file not found. Returning None.")
            return None
        except Exception as e:
            logger.error(f"Error reading last training time: {e}")
            return None

    def train(self, input_data) -> bool:
        """
        Train the Random Forest model on provided data.

        Args:
            input_data: Path to CSV file or pandas DataFrame with raw data.
        Returns:
            bool: True if training succeeded, False otherwise.
        """
        logger.info("Starting RF training")
        # Prepare raw input
        try:
            if isinstance(input_data, pd.DataFrame):
                df_raw = input_data.copy()
                temp_raw = os.path.join(self.models_dir, "temp_raw.csv")
                df_raw.to_csv(temp_raw, index=False)
                raw_path = temp_raw
            else:
                raw_path = input_data
                if not os.path.exists(raw_path):
                    logger.error(f"Input file not found: {raw_path}")
                    return False
            logger.info(f"Loaded raw data from {raw_path}")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

        # Process OT labels
        try:
            processed_csv = os.path.join(self.models_dir, "processed.csv")
            ot_quantile = config.get("ml", "ot_quantile", fallback=0.65)
            processor = OptimalThroughputProcessor(
                input_csv=raw_path,
                output_csv=processed_csv,
                quantile=ot_quantile
            )
            processor.run()
            df = pd.read_csv(processed_csv)
            logger.info(f"Processed data for OT, shape={df.shape}")
        except Exception as e:
            logger.error(f"Error in OT processing: {e}")
            return False

        # Ensure enough samples
        min_samples = config.get("ml", "min_training_samples")
        if len(df) < min_samples:
            logger.warning("Not enough samples for training. Need at least %d.", min_samples)
            return False

        # Prepare features and target
        try:
            # Numeric columns except labels
            numeric = df.select_dtypes(include=['number']).copy()
            feature_cols = [c for c in numeric.columns if c not in ['OT', 'throughput_KBps']]
            X = numeric[feature_cols]
            y = numeric['OT']

            # Split
            test_size = config.get("ml", "test_size", fallback=0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            n_estimators = config.get("ml", "n_estimators", fallback=100)
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            self.model.fit(X_train, y_train)
            logger.info("Random Forest trained")

            # Evaluate
            preds = self.model.predict(X_test)
            probs = self.model.predict_proba(X_test)[:, 1]
            logger.info(f"Accuracy={accuracy_score(y_test, preds):.4f}, ROC AUC={roc_auc_score(y_test, probs):.4f}")
            logger.info("Report:\n%s", classification_report(y_test, preds))

            # Save
            model_path = os.path.join(self.models_dir, "rf_model.joblib")
            joblib.dump(self.model, model_path)
            self.set_last_training_time()
            logger.info(f"Model saved to {model_path}")

            # Cleanup
            for f in [locals().get('temp_raw'), processed_csv]:
                if f and os.path.exists(f):
                    os.remove(f)
            return True
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def load(self) -> bool:
        """
        Load trained RF model from disk.
        """
        try:
            path = os.path.join(self.models_dir, "rf_model.joblib")
            if not os.path.exists(path):
                logger.warning("Model file not found")
                return False
            self.model = joblib.load(path)
            logger.info("Model loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def find_optimal_chunk_size(model, row, candidate_chunks, tolerance=1e-4):
        """
        Iterate over candidate chunk sizes to maximize P(OT=1).
        """
        probs = []
        for c in candidate_chunks:
            feat = [row['file_size_KB'], c, row['access_count'], row['access_count_label']]
            p = model.predict_proba([feat])[:, 1][0]
            probs.append(p)
        arr = np.array(probs)
        max_prob = arr.max()
        idxs = np.where(np.abs(arr - max_prob) < tolerance)[0]
        return int(candidate_chunks[idxs].max()), float(max_prob)

    def predict(self, df_raw) -> pd.DataFrame:
        """
        Predict optimal chunk sizes for incoming raw data.
        """
        if self.model is None and not self.load():
            logger.error("Model not loaded")
            return None

        # Write raw and process OT
        try:
            temp = os.path.join(self.models_dir, "temp_pred.csv")
            df_raw.to_csv(temp, index=False)
            proc = OptimalThroughputProcessor(temp, temp, quantile=None)
            proc.run()
            df = pd.read_csv(temp)
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None

        # Predictions
        try:
            cand = np.sort(df['chunk_size_KB'].unique())
            recs = []
            for _, row in df.iterrows():
                opt, prob = self.find_optimal_chunk_size(self.model, row, cand)
                recs.append({
                    'file_path': row.get('file_path', ''),
                    'file_size_KB': row['file_size_KB'],
                    'current_chunk_KB': row['chunk_size_KB'],
                    'optimal_chunk_KB': opt,
                    'probability': prob
                })
            return pd.DataFrame(recs)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
        finally:
            if os.path.exists(temp):
                os.remove(temp)
            if os.path.exists(os.path.join(self.models_dir, "processed.csv")):
                os.remove(os.path.join(self.models_dir, "processed.csv"))
            logger.info("Temporary files cleaned up")
            return None