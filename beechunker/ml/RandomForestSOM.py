import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from beechunker.common.beechunker_logging import setup_logging
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from beechunker.common.config import config
from beechunker.ml.feature_extraction import OptimalThroughputProcessor

logger = setup_logging("rf_classifier")

class BeeChunkerRF:
    """
    Stacked Random Forest + Gradient Boosting classifier for OT prediction
    and subsequent optimal chunk size selection.
    """
    def __init__(self):
        """Initialize the ensemble model and paths."""
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
        Train a stacking ensemble: RandomForest + HistGradientBoosting,
        final estimator LogisticRegression.

        Args:
            input_data: CSV filepath or DataFrame of raw logs.
        Returns:
            bool: training success.
        """
        logger.info("Starting stacked RF training")
        # load raw
        try:
            if isinstance(input_data, pd.DataFrame):
                df_raw = input_data.copy()
            else:
                df_raw = pd.read_csv(input_data)
            logger.info(f"Loaded raw data, shape={df_raw.shape}")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

        # OT label processing
        try:
            ot_quantile = config.get("ml", "ot_quantile", fallback=0.65)
            tmp_in = os.path.join(self.models_dir, "_tmp_raw.csv")
            df_raw.to_csv(tmp_in, index=False)
            tmp_out = os.path.join(self.models_dir, "_tmp_proc.csv")
            proc = OptimalThroughputProcessor(tmp_in, tmp_out, quantile=ot_quantile)
            proc.run()
            df = pd.read_csv(tmp_out)
            logger.info(f"Processed OT labels, shape={df.shape}")
        except Exception as e:
            logger.error(f"Error in OT processing: {e}")
            return False
        finally:
            for f in [tmp_in]:
                if os.path.exists(f): os.remove(f)

        # require minimum samples
        min_s = config.get("ml", "min_training_samples")
        if len(df) < min_s:
            logger.warning("Not enough samples (%d) < min required %d", len(df), min_s)
            return False

        # features/target
        num = df.select_dtypes(include=[np.number])
        X = num.drop(columns=["OT", "throughput_KBps"])  # all numeric except labels
        y = num["OT"]

        # train/test split
        test_size = config.get("ml", "test_size", fallback=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # base learners
        n_est = config.get("ml", "n_estimators", fallback=100)
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=config.get("ml", "hgb_iter", fallback=20),
            random_state=42
        )

        rf.fit(X_train, y_train)
        hgb.fit(X_train, y_train)
        logger.info("Trained RF and HGB base learners")

        # stacking ensemble
        stack = StackingClassifier(
            estimators=[('rf', rf), ('hgb', hgb)],
            final_estimator=LogisticRegression(
                class_weight='balanced', solver='liblinear', max_iter=1000
            ),
            cv='prefit'
        )
        stack.fit(X_train, y_train)
        self.model = stack
        logger.info("Stacking classifier trained")

        # eval
        preds = stack.predict(X_test)
        probs = stack.predict_proba(X_test)[:, 1]
        logger.info(
            f"Test Accuracy={accuracy_score(y_test, preds):.4f}, "
            f"ROC AUC={roc_auc_score(y_test, probs):.4f}"
        )
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))

        # persist
        mpath = os.path.join(self.models_dir, "rf_model.joblib")
        joblib.dump(self.model, mpath)
        self.set_last_training_time()
        logger.info(f"Stacked model saved to {mpath}")

        # cleanup
        if os.path.exists(tmp_out): os.remove(tmp_out)
        return True

    def load(self) -> bool:
        """
        Load the trained stacking model from disk.
        """
        path = os.path.join(self.models_dir, "rf_model.joblib")
        if not os.path.exists(path):
            logger.warning("Model file not found at %s", path)
            return False
        try:
            self.model = joblib.load(path)
            logger.info("Model loaded from %s", path)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def find_optimal_chunk_size(model, row, candidate_chunks, tolerance=1e-4):
        """
        For each candidate chunk size, compute P(OT=1) and choose the size
        with the highest probability (breaking ties by larger chunk).
        """
        probs = []
        for c in candidate_chunks:
            feat = [row['file_size_KB'], c, row['access_count'], row['access_count_label']]
            probs.append(model.predict_proba([feat])[0, 1])
        arr = np.array(probs)
        idxs = np.where(np.abs(arr - arr.max()) < tolerance)[0]
        return int(candidate_chunks[idxs].max()), float(arr.max())

    def predict(self, df_raw):
        """
        Predict optimal chunk sizes for incoming raw DataFrame.
        Returns a DataFrame of predictions.
        """
        if self.model is None and not self.load():
            logger.error("Model not loaded. Cannot predict.")
            return None

        # preprocess OT labels
        tmp = os.path.join(self.models_dir, "temp_pred.csv")
        df_raw.to_csv(tmp, index=False)
        proc = OptimalThroughputProcessor(tmp, tmp, quantile=config.get("ml", "ot_quantile", fallback=0.65))
        proc.run()
        df = pd.read_csv(tmp)

        # cleanup
        os.remove(tmp)
        if os.path.exists(os.path.join(self.models_dir, "_tmp_proc.csv")):
            os.remove(os.path.join(self.models_dir, "_tmp_proc.csv"))

        # generate predictions
        cand = np.sort(df['chunk_size_KB'].unique())
        records = []
        for _, row in df.iterrows():
            opt, prob = self.find_optimal_chunk_size(self.model, row, cand)
            records.append({
                'file_path': row.get('file_path', ''),
                'file_size_KB': row['file_size_KB'],
                'current_chunk_KB': row['chunk_size_KB'],
                'optimal_chunk_KB': opt,
                'probability': prob
            })
        return pd.DataFrame(records)
