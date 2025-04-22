# import os
# import numpy as np
# import pandas as pd
# import joblib
# from datetime import datetime
# from beechunker.common.beechunker_logging import setup_logging
# from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# from xgboost import XGBClassifier

# from beechunker.common.config import config
# from beechunker.ml.feature_extraction import OptimalThroughputProcessor

# logger = setup_logging("xgb_classifier")


# class BeeChunkerXGB:
#     """
#     Stacked XGBoost + HistGradientBoosting classifier for OT prediction
#     and subsequent optimal chunk size selection.
#     """

#     def __init__(self):
#         """Initialize the ensemble model and paths, with a writable fallback."""
#         self.model = None
#         self.feature_names = None
#         cfg_dir = config.get("ml", "models_dir")
#         try:
#             os.makedirs(cfg_dir, exist_ok=True)
#             self.models_dir = cfg_dir
#         except PermissionError:
#             fallback = os.path.join(os.getcwd(), "models")
#             os.makedirs(fallback, exist_ok=True)
#             self.models_dir = fallback
#             logger.warning(
#                 "Cannot write to configured models_dir '%s'. Falling back to '%s'.",
#                 cfg_dir,
#                 fallback,
#             )

#     def set_last_training_time(self):
#         path = os.path.join(self.models_dir, "xgb_last_training_time.txt")
#         with open(path, "w") as f:
#             f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

#     def get_last_training_time(self):
#         path = os.path.join(self.models_dir, "xgb_last_training_time.txt")
#         try:
#             with open(path, "r") as f:
#                 ts = f.read().strip()
#             return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
#         except FileNotFoundError:
#             logger.warning("Last training time file not found. Returning None.")
#             return None
#         except Exception as e:
#             logger.error(f"Error reading last training time: {e}")
#             return None

#     def train(self, input_data) -> bool:
#         """
#         Train a stacking ensemble: XGBoost + HistGradientBoosting,
#         final estimator LogisticRegression.
#         """
#         logger.info("Starting stacked XGB training")
#         # load raw
#         try:
#             if isinstance(input_data, pd.DataFrame):
#                 df_raw = input_data.copy()
#             else:
#                 df_raw = pd.read_csv(input_data)
#             logger.info(f"Loaded raw data, shape={df_raw.shape}")
#         except Exception as e:
#             logger.error(f"Error loading raw data: {e}")
#             return False

#         # OT label processing
#         tmp_in = os.path.join(self.models_dir, "xgb_tmp_raw.csv")
#         tmp_out = os.path.join(self.models_dir, "xgb_tmp_proc.csv")
#         try:
#             ot_quantile = config.get("ml", "ot_quantile")
#             df_raw.to_csv(tmp_in, index=False)
#             proc = OptimalThroughputProcessor(tmp_in, tmp_out, quantile=ot_quantile)
#             proc.run()
#             df = pd.read_csv(tmp_out)
#             logger.info(f"Processed OT labels, shape={df.shape}")
#         except Exception as e:
#             logger.error(f"Error in OT processing: {e}")
#             return False
#         finally:
#             if os.path.exists(tmp_in):
#                 os.remove(tmp_in)

#         # require minimum samples
#         min_s = config.get("ml", "min_training_samples")
#         if len(df) < min_s:
#             logger.warning("Not enough samples (%d) < min required %d", len(df), min_s)
#             return False

#         # features/target
#         num = df.select_dtypes(include=[np.number])
#         X = num.drop(columns=["OT", "throughput_KBps"])
#         y = num["OT"]

#         self.feature_names = X.columns.tolist()
#         candidate_chunks = sorted(df['chunk_size_KB'].unique())
#         joblib.dump(candidate_chunks, os.path.join(self.models_dir, "xgb_candidate_chunks.joblib"))
#         self.candidate_chunks = candidate_chunks

#         # train/test split
#         test_size = config.get("ml", "test_size")
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, stratify=y, random_state=42
#         )

#         # base learners
#         n_est = config.get("ml", "n_estimators")
#         xgb = XGBClassifier(
#             n_estimators=n_est,
#             use_label_encoder=False,
#             eval_metric="logloss",
#             random_state=42,
#             n_jobs=-1,
#         )
#         hgb = HistGradientBoostingClassifier(
#             max_iter=config.get("ml", "hgb_iter"), random_state=42
#         )

#         xgb.fit(X_train, y_train)
#         hgb.fit(X_train, y_train)
#         logger.info("Trained XGBoost and HGB base learners")

#         # stacking ensemble
#         stack = StackingClassifier(
#             estimators=[("xgb", xgb), ("hgb", hgb)],
#             final_estimator=LogisticRegression(
#                 class_weight="balanced", solver="liblinear", max_iter=1000
#             ),
#             cv="prefit",
#         )
#         stack.fit(X_train, y_train)
#         self.model = stack
#         logger.info("Stacking classifier trained")

#         # evaluation
#         preds = stack.predict(X_test)
#         probs = stack.predict_proba(X_test)[:, 1]
#         logger.info(
#             f"Test Accuracy={accuracy_score(y_test, preds):.4f}, "
#             f"ROC AUC={roc_auc_score(y_test, probs):.4f}"
#         )
#         logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))

#         # persist ensemble and components
#         ensemble_path = os.path.join(self.models_dir, "xgb_model.joblib")
#         joblib.dump(self.model, ensemble_path)
#         logger.info(f"Stacked model saved to {ensemble_path}")

#         xgb_path = os.path.join(self.models_dir, "xgb_base.joblib")
#         hgb_path = os.path.join(self.models_dir, "xgb_hgb_base.joblib")
#         joblib.dump(xgb, xgb_path)
#         joblib.dump(hgb, hgb_path)
#         logger.info(f"XGBoost base learner saved to {xgb_path}")
#         logger.info(f"HGB base learner saved to {hgb_path}")

#         meta_path = os.path.join(self.models_dir, "xgb_logistic_meta.joblib")
#         joblib.dump(self.model.final_estimator_, meta_path)
#         logger.info(f"Logistic meta‑learner saved to {meta_path}")

#         fnames_path = os.path.join(self.models_dir, "xgb_feature_names.joblib")
#         joblib.dump(self.feature_names, fnames_path)
#         self.set_last_training_time()

#         if os.path.exists(tmp_out):
#             os.remove(tmp_out)

#         return True

#     def load(self) -> bool:
#         """
#         Load the trained stacking model from disk.
#         """
#         path = os.path.join(self.models_dir, "xgb_model.joblib")
#         if not os.path.exists(path):
#             logger.warning("Model file not found at %s", path)
#             return False
#         try:
#             self.model = joblib.load(path)
#             logger.info("Model loaded from %s", path)
#             self.feature_names = joblib.load(
#                 os.path.join(self.models_dir, "xgb_feature_names.joblib")
#             )
#             self.candidate_chunks = joblib.load(
#                 os.path.join(self.models_dir, "xgb_candidate_chunks.joblib")
#             )
#             return True
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             return False

#     @staticmethod
#     def find_optimal_chunk_size(model, row, feature_names, candidate_chunks, tolerance=1e-4):
#         # Only consider chunks smaller than the file size
#         valid_chunks = [c for c in candidate_chunks if c < row.get('file_size_KB', np.inf)]
#         if not valid_chunks:
#             valid_chunks = candidate_chunks

#         probs = []
#         for c in valid_chunks:
#             rc = row.copy()
#             rc['chunk_size_KB'] = c
#             arr = rc[feature_names].values.reshape(1, -1)
#             p = model.predict_proba(arr)[0, 1]
#             probs.append(p)

#         max_prob = max(probs)
#         best_chunks = [valid_chunks[i] for i, p in enumerate(probs) if abs(p - max_prob) < tolerance]
#         return int(max(best_chunks)), float(max_prob)

#     def predict(self, df_raw):
#         if self.model is None and not self.load():
#             logger.error("Model not loaded. Cannot predict.")
#             return None

#         df_input = df_raw.copy()
#         # … (same preprocessing as before) …

#         tmp = os.path.join(self.models_dir, "xgb_temp_pred.csv")
#         df_input.to_csv(tmp, index=False)
#         proc = OptimalThroughputProcessor(tmp, tmp, quantile=config.get("ml", "ot_quantile"))
#         proc.run()

#         df = pd.read_csv(tmp)
#         os.remove(tmp)
#         tmp_proc = os.path.join(self.models_dir, "xgb_tmp_proc.csv")
#         if os.path.exists(tmp_proc):
#             os.remove(tmp_proc)

#         records = []
#         for _, row in df.iterrows():
#             opt, prob = self.find_optimal_chunk_size(
#                 self.model, row, self.feature_names, self.candidate_chunks
#             )
#             records.append({
#                 'file_path': row.get('file_path', ''),
#                 'file_size_KB': row['file_size_KB'],
#                 'current_chunk_KB': row['chunk_size_KB'],
#                 'optimal_chunk_KB': opt,
#                 'confidance': prob
#             })

#         # Print all records
#         print(f"Predictions: {len(records)} records")
#         for rec in records:
#             print("\n")
#             print(f"File: {rec['file_size_KB']} KB, \nCurrent Chunk: {rec['current_chunk_KB']} KB, \n"
#                   f"Optimal Chunk: {rec['optimal_chunk_KB']} KB, \nConfidence: {rec['confidance']:.4f}")

#         print("\n------------------------------------------ \n")

#         df_preds = pd.DataFrame(records)
#         best_idx = df_preds['confidance'].idxmax()
#         # Return the best optimal chunk size (always < file size)
#         return int(df_preds.loc[best_idx, 'optimal_chunk_KB'])




import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from beechunker.common.beechunker_logging import setup_logging
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier

from beechunker.common.config import config
from beechunker.ml.feature_extraction import OptimalThroughputProcessor

logger = setup_logging("xgb_classifier")


class BeeChunkerXGB:
    """
    Stacked XGBoost + HistGradientBoosting classifier for OT prediction
    and subsequent optimal chunk size selection based on expected throughput.
    """

    def __init__(self):
        """Initialize the ensemble model and paths, with a writable fallback."""
        self.model = None
        self.feature_names = None
        cfg_dir = config.get("ml", "models_dir")
        try:
            os.makedirs(cfg_dir, exist_ok=True)
            self.models_dir = cfg_dir
        except PermissionError:
            fallback = os.path.join(os.getcwd(), "models")
            os.makedirs(fallback, exist_ok=True)
            self.models_dir = fallback
            logger.warning(
                "Cannot write to configured models_dir '%s'. Falling back to '%s'.",
                cfg_dir,
                fallback,
            )

    def set_last_training_time(self):
        path = os.path.join(self.models_dir, "xgb_last_training_time.txt")
        with open(path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_last_training_time(self):
        path = os.path.join(self.models_dir, "xgb_last_training_time.txt")
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
        Train a stacking ensemble: XGBoost + HistGradientBoosting,
        final estimator LogisticRegression.
        """
        logger.info("Starting stacked XGB training")
        try:
            df_raw = input_data.copy() if isinstance(input_data, pd.DataFrame) else pd.read_csv(input_data)
            logger.info(f"Loaded raw data, shape={df_raw.shape}")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

        # OT label processing
        tmp_in = os.path.join(self.models_dir, "xgb_tmp_raw.csv")
        tmp_out = os.path.join(self.models_dir, "xgb_tmp_proc.csv")
        try:
            df_raw.to_csv(tmp_in, index=False)
            proc = OptimalThroughputProcessor(tmp_in, tmp_out, quantile=config.get("ml", "ot_quantile"))
            proc.run()
            df = pd.read_csv(tmp_out)
            logger.info(f"Processed OT labels, shape={df.shape}")
        except Exception as e:
            logger.error(f"Error in OT processing: {e}")
            return False
        finally:
            if os.path.exists(tmp_in): os.remove(tmp_in)

        if len(df) < config.get("ml", "min_training_samples"):
            logger.warning("Not enough samples for training.")
            return False

        num = df.select_dtypes(include=[np.number])
        X = num.drop(columns=["OT", "throughput_KBps"])
        y = num["OT"]

        self.feature_names = X.columns.tolist()
        candidate_chunks = sorted(df['chunk_size_KB'].unique())
        joblib.dump(candidate_chunks, os.path.join(self.models_dir, "xgb_candidate_chunks.joblib"))
        self.candidate_chunks = candidate_chunks

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.get("ml", "test_size"), stratify=y, random_state=42
        )

        xgb = XGBClassifier(
            n_estimators=config.get("ml", "n_estimators"),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=config.get("ml", "hgb_iter"), random_state=42
        )

        xgb.fit(X_train, y_train)
        hgb.fit(X_train, y_train)
        logger.info("Trained XGBoost and HGB base learners")

        stack = StackingClassifier(
            estimators=[("xgb", xgb), ("hgb", hgb)],
            final_estimator=LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000),
            cv="prefit",
        )
        stack.fit(X_train, y_train)
        self.model = stack

        preds = stack.predict(X_test)
        probs = stack.predict_proba(X_test)[:, 1]
        logger.info(f"Test Accuracy={accuracy_score(y_test, preds):.4f}, ROC AUC={roc_auc_score(y_test, probs):.4f}")
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))

        # Persist
        joblib.dump(self.model, os.path.join(self.models_dir, "xgb_model.joblib"))
        joblib.dump(xgb, os.path.join(self.models_dir, "xgb_base.joblib"))
        joblib.dump(hgb, os.path.join(self.models_dir, "xgb_hgb_base.joblib"))
        joblib.dump(self.model.final_estimator_, os.path.join(self.models_dir, "xgb_logistic_meta.joblib"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "xgb_feature_names.joblib"))
        self.set_last_training_time()
        if os.path.exists(tmp_out): os.remove(tmp_out)
        return True

    def load(self) -> bool:
        path = os.path.join(self.models_dir, "xgb_model.joblib")
        if not os.path.exists(path):
            logger.warning("Model file not found at %s", path)
            return False
        try:
            self.model = joblib.load(path)
            self.feature_names = joblib.load(os.path.join(self.models_dir, "xgb_feature_names.joblib"))
            self.candidate_chunks = joblib.load(os.path.join(self.models_dir, "xgb_candidate_chunks.joblib"))
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def find_optimal_chunk_size(model, row, feature_names, candidate_chunks, tolerance=1e-4):
        # only sizes < file size
        valid_chunks = [c for c in candidate_chunks if c < row.get('file_size_KB', np.inf)]
        if not valid_chunks: valid_chunks = candidate_chunks

        probs, exp_tps = [], []
        for c in valid_chunks:
            rc = row.copy(); rc['chunk_size_KB'] = c
            x = rc[feature_names].values.reshape(1, -1)
            p = model.predict_proba(x)[0, 1]
            probs.append(p); exp_tps.append(c * p)

        idx = int(np.argmax(exp_tps))
        return int(valid_chunks[idx]), float(exp_tps[idx]), float(probs[idx])

    def predict(self, df_raw):
        if self.model is None and not self.load():
            logger.error("Model not loaded. Cannot predict.")
            return None

        df_input = df_raw.copy()
        tmp = os.path.join(self.models_dir, "xgb_temp_pred.csv")
        df_input.to_csv(tmp, index=False)
        proc = OptimalThroughputProcessor(tmp, tmp, quantile=config.get("ml", "ot_quantile"))
        proc.run()
        df = pd.read_csv(tmp); os.remove(tmp)
        if os.path.exists(os.path.join(self.models_dir, "xgb_tmp_proc.csv")):
            os.remove(os.path.join(self.models_dir, "xgb_tmp_proc.csv"))

        records = []
        for _, row in df.iterrows():
            opt, exp_tp, conf = self.find_optimal_chunk_size(
                self.model, row, self.feature_names, self.candidate_chunks
            )
            records.append({
                'file_path': row.get('file_path', ''),
                'file_size_KB': row['file_size_KB'],
                'current_chunk_KB': row['chunk_size_KB'],
                'optimal_chunk_KB': opt,
                'expected_throughput': exp_tp,
                'confidence': conf
            })

        print(f"Predictions: {len(records)} records")
        for rec in records:
            print(
                f"File: {rec['file_size_KB']}, Current Chunk: {rec['current_chunk_KB']} KB, "
                f"Optimal Chunk: {rec['optimal_chunk_KB']} KB, "
                f"Expected Throughput: {rec['expected_throughput']:.2f} KB/s, "
                f"Confidence: {rec['confidence']:.4f}"
            )
        print("\n------------------------------------------ \n")

        df_preds = pd.DataFrame(records)
        best_idx = df_preds['expected_throughput'].idxmax()
        return int(df_preds.loc[best_idx, 'optimal_chunk_KB'])

