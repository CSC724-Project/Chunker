import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from beechunker.common.beechunker_logging import setup_logging
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from beechunker.common.config import config
from beechunker.ml.feature_extraction import OptimalThroughputProcessor

logger = setup_logging("rf_classifier")


class BeeChunkerRF:
    """
    Stacked Random Forest + Gradient Boosting classifier for OT prediction
    and subsequent optimal chunk size selection.
    """

    def __init__(self):
        """Initialize the ensemble model and paths, with a writable fallback."""
        self.model = None
        self.feature_names = None
        cfg_dir = config.get("ml", "models_dir")
        # Try the configured path first
        try:
            os.makedirs(cfg_dir, exist_ok=True)
            self.models_dir = cfg_dir
        except PermissionError:
            # Fallback to a local 'models' directory under cwd
            fallback = os.path.join(os.getcwd(), "models")
            os.makedirs(fallback, exist_ok=True)
            self.models_dir = fallback
            logger.warning(
                "Cannot write to configured models_dir '%s'. " "Falling back to '%s'.",
                cfg_dir,
                fallback,
            )

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
        tmp_in = None
        try:
            ot_quantile = config.get("ml", "ot_quantile")
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
                if os.path.exists(f):
                    os.remove(f)

        # require minimum samples
        min_s = config.get("ml", "min_training_samples")
        if len(df) < min_s:
            logger.warning("Not enough samples (%d) < min required %d", len(df), min_s)
            return False

        # features/target
        num = df.select_dtypes(include=[np.number])
        X = num.drop(columns=["OT", "throughput_KBps"])  # all numeric except labels
        y = num["OT"]

        # save feature list
        self.feature_names = X.columns.tolist()

        # save candidate chunk sizes (global list)
        candidate_chunks = sorted(df['chunk_size_KB'].unique())
        joblib.dump(candidate_chunks, os.path.join(self.models_dir, "candidate_chunks.joblib"))

        # keep it in memory too, so predict() works right away
        self.candidate_chunks = candidate_chunks

        # train/test split
        test_size = config.get("ml", "test_size")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # base learners
        n_est = config.get("ml", "n_estimators")
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=config.get("ml", "hgb_iter"), random_state=42
        )

        rf.fit(X_train, y_train)
        hgb.fit(X_train, y_train)
        logger.info("Trained RF and HGB base learners")

        # stacking ensemble
        stack = StackingClassifier(
            estimators=[("rf", rf), ("hgb", hgb)],
            final_estimator=LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=1000
            ),
            cv="prefit",
        )
        stack.fit(X_train, y_train)
        self.model = stack
        logger.info("Stacking classifier trained")

        # eval
        preds = stack.predict(X_test)
        probs = stack.predict_proba(X_test)[:, 1]

        # compute metrics
        acc     = accuracy_score(y_test, preds)
        prec    = precision_score(y_test, preds, average='binary')
        rec     = recall_score(y_test, preds, average='binary')
        f1      = f1_score(y_test, preds, average='binary')
        roc_auc = roc_auc_score(y_test, probs)

        # log them all in one line
        logger.info(
            f"Test Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}"
        )

        # and the detailed breakdown
        logger.info("Classification Report:\n%s", classification_report(y_test, preds))
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))


        # persist full ensemble
        ensemble_path = os.path.join(self.models_dir, "rf_model.joblib")
        joblib.dump(self.model, ensemble_path)
        logger.info(f"Stacked model saved to {ensemble_path}")

        # persist base learners
        rf_path = os.path.join(self.models_dir, "rf_base.joblib")
        hgb_path = os.path.join(self.models_dir, "hgb_base.joblib")
        joblib.dump(rf, rf_path)
        joblib.dump(hgb, hgb_path)
        logger.info(f"RF base learner saved to {rf_path}")
        logger.info(f"HGB base learner saved to {hgb_path}")

        # persist meta‑learner
        meta_path = os.path.join(self.models_dir, "logistic_meta.joblib")
        joblib.dump(self.model.final_estimator_, meta_path)
        logger.info(f"Logistic meta‑learner saved to {meta_path}")

        # persist feature names
        joblib.dump(
            self.feature_names, os.path.join(self.models_dir, "feature_names.joblib")
        )

        # record training time
        self.set_last_training_time()

        # cleanup
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

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
            # reload feature names and chunk list
            self.feature_names = joblib.load(
                os.path.join(self.models_dir, "feature_names.joblib")
            )
            self.candidate_chunks = joblib.load(
                os.path.join(self.models_dir, "candidate_chunks.joblib")
            )
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def find_optimal_chunk_size(model, row, feature_names, candidate_chunks, tolerance=1e-4):
        """
        For each candidate chunk size, compute P(OT=1) and choose the size
        with the highest probability (breaking ties by larger chunk).
        """
        probs = []
        # build a probability for each possible chunk size
        for c in candidate_chunks:
            rc = row.copy()
            rc['chunk_size_KB'] = c
            # construct the feature vector in the exact order used at training
            x = rc[feature_names].values.reshape(1, -1)
            p = model.predict_proba(x)[0, 1]
            probs.append(p)

        arr = np.array(probs)
        max_prob = arr.max()

        # collect all chunks within tolerance of the maximum probability
        best_chunks = [
            candidate_chunks[i]
            for i, p in enumerate(arr)
            if abs(p - max_prob) < tolerance
        ]

        # pick the largest chunk size among the ties
        optimal_chunk = max(best_chunks)
        return int(optimal_chunk), float(max_prob)

    def predict(self, df_raw):
        """
        Predict optimal chunk sizes for incoming raw DataFrame.
        Returns a DataFrame of predictions.
        """
        if self.model is None and not self.load():
            logger.error("Model not loaded. Cannot predict.")
            return None

        # Create a copy to avoid modifying the original
        df_input = df_raw.copy()

        # Ensure file_size_KB exists
        if 'file_size_KB' not in df_input.columns and 'file_size' in df_input.columns:
            df_input['file_size_KB'] = df_input['file_size'] / 1024
        
        # Ensure chunk_size_KB exists
        if 'chunk_size_KB' not in df_input.columns and 'chunk_size' in df_input.columns:
            df_input['chunk_size_KB'] = df_input['chunk_size'] / 1024
        elif 'chunk_size_KB' not in df_input.columns:
            df_input['chunk_size_KB'] = 512  # Default chunk size in KB
        
        # Convert throughput_mbps to throughput_KBps if present
        if 'throughput_mbps' in df_input.columns:
            df_input['throughput_KBps'] = df_input['throughput_mbps'] * 1024  # Convert MB/s to KB/s
        elif 'throughput_KBps' not in df_input.columns:
            df_input['throughput_KBps'] = 100 * 1024  # Default to 100 MB/s in KB/s

        # Map existing features to expected feature names
        feature_mapping = {
            'avg_read_size': 'avg_read_KB',
            'avg_write_size': 'avg_write_KB',
            'max_read_size': 'max_read_KB',
            'max_write_size': 'max_write_KB',
            'read_count': 'read_ops',
            'write_count': 'write_ops'
        }
        
        for old_name, new_name in feature_mapping.items():
            if old_name in df_input.columns:
                df_input[new_name] = df_input[old_name] / 1024 if 'size' in old_name else df_input[old_name]
        
        # preprocess OT labels
        tmp = os.path.join(self.models_dir, "temp_pred.csv")
        
        df_input.to_csv(tmp, index=False)
        proc = OptimalThroughputProcessor(tmp, tmp, quantile=config.get("ml", "ot_quantile"))
        proc.run()
        df = pd.read_csv(tmp)

        # cleanup
        os.remove(tmp)
        if os.path.exists(os.path.join(self.models_dir, "_tmp_proc.csv")):
            os.remove(os.path.join(self.models_dir, "_tmp_proc.csv"))

        # generate predictions using the full set of trained chunk sizes
        cand = self.candidate_chunks
        records = []
        for _, row in df.iterrows():
            opt, prob = self.find_optimal_chunk_size(self.model, row, self.feature_names, cand)
            records.append({
                'file_path': row.get('file_path', ''),
                'file_size_KB': row['file_size_KB'],
                'current_chunk_KB': row['chunk_size_KB'],
                'optimal_chunk_KB': opt,
                'confidance': prob
            })

        
        # convert to DataFrame
        df_preds = pd.DataFrame(records)

        # pick only the one with highest confidence
        best_idx = df_preds['confidance'].idxmax()
        best_row = df_preds.loc[[best_idx]].reset_index(drop=True)
        best_chunk  = int(df_preds.loc[best_idx, 'optimal_chunk_KB'])

        return best_chunk