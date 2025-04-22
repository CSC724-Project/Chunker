import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from beechunker.common.beechunker_logging import setup_logging
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from beechunker.common.config import config

logger = setup_logging("lr_regressor")


class BeeChunkerLR:
    """
    Linear Regression pipeline (impute + regression) to predict throughput and select
    optimal chunk size based on maximizing predicted throughput.
    """

    def __init__(self):
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
        path = os.path.join(self.models_dir, "lr_last_training_time.txt")
        with open(path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_last_training_time(self):
        path = os.path.join(self.models_dir, "lr_last_training_time.txt")
        try:
            with open(path, "r") as f:
                ts = f.read().strip()
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            logger.warning("Training timestamp not available.")
            return None

    def train(self, input_data) -> bool:
        """
        Train a Linear Regression pipeline on raw logs to predict throughput.
        """
        logger.info("Starting LR training")
        try:
            df_raw = input_data.copy() if isinstance(input_data, pd.DataFrame) else pd.read_csv(input_data)
            logger.info(f"Loaded raw data, shape={df_raw.shape}")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

        # Preprocess
        df = df_raw.copy()
        if 'file_size_KB' not in df.columns and 'file_size' in df.columns:
            df['file_size_KB'] = df['file_size'] / 1024
        if 'chunk_size_KB' not in df.columns:
            if 'chunk_size' in df.columns:
                df['chunk_size_KB'] = df['chunk_size'] / 1024
            else:
                df['chunk_size_KB'] = df['file_size_KB'] / 2
        if 'throughput_mbps' in df.columns and 'throughput_KBps' not in df.columns:
            df['throughput_KBps'] = df['throughput_mbps'] * 1024

        mapping = {
            'avg_read_size': 'avg_read_KB',
            'avg_write_size': 'avg_write_KB',
            'max_read_size': 'max_read_KB',
            'max_write_size': 'max_write_KB',
            'read_count': 'read_ops',
            'write_count': 'write_ops'
        }
        for old, new in mapping.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old] / (1024 if 'size' in old else 1)

        # Numeric features
        num = df.select_dtypes(include=[np.number])
        # Drop any numeric columns that are all NaN (e.g., 'error_message')
        num = num.dropna(axis=1, how='all')
        if 'throughput_KBps' not in num.columns:
            logger.error("No throughput_KBps column after preprocessing.")
            return False
        X = num.drop(columns=['throughput_KBps'])
        y = num['throughput_KBps']

        self.feature_names = X.columns.tolist()
        candidate_chunks = sorted(df['chunk_size_KB'].unique())
        joblib.dump(candidate_chunks, os.path.join(self.models_dir, "lr_candidate_chunks.joblib"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "lr_feature_names.joblib"))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.get("ml", "test_size"), random_state=42
        )

        # Pipeline: impute missing then regression
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('lr', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        self.model = pipeline

        preds = pipeline.predict(X_test)
        # Compute RMSE manually to avoid sklearn version issues
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        logger.info(
            f"LR RMSE={rmse:.4f}, R2={r2_score(y_test, preds):.4f}"
        )

        # Persist pipeline
        joblib.dump(pipeline, os.path.join(self.models_dir, "lr_model.joblib"))
        self.set_last_training_time()
        return True

    def load(self) -> bool:
        path = os.path.join(self.models_dir, "lr_model.joblib")
        if not os.path.exists(path):
            logger.warning(f"Model file not found at {path}")
            return False
        try:
            self.model = joblib.load(path)
            self.feature_names = joblib.load(os.path.join(self.models_dir, "lr_feature_names.joblib"))
            self.candidate_chunks = joblib.load(os.path.join(self.models_dir, "lr_candidate_chunks.joblib"))
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def find_optimal_chunk_size(model, row, feature_names, candidate_chunks):
        valid_chunks = [c for c in candidate_chunks if c < row.get('file_size_KB', np.inf)]
        if not valid_chunks:
            valid_chunks = candidate_chunks

        preds = []
        for c in valid_chunks:
            rc = row.copy()
            rc['chunk_size_KB'] = c
            x = rc[feature_names].values.reshape(1, -1)
            preds.append(model.predict(x)[0])

        idx = int(np.argmax(preds))
        return int(valid_chunks[idx]), float(preds[idx])

    def predict(self, df_raw):
        if self.model is None and not self.load():
            logger.error("Model not loaded. Cannot predict.")
            return None

        df = df_raw.copy()
        if 'file_size_KB' not in df.columns and 'file_size' in df.columns:
            df['file_size_KB'] = df['file_size'] / 1024
        if 'chunk_size_KB' not in df.columns:
            if 'chunk_size' in df.columns:
                df['chunk_size_KB'] = df['chunk_size'] / 1024
            else:
                df['chunk_size_KB'] = df['file_size_KB'] / 2
        mapping = {
            'avg_read_size': 'avg_read_KB',
            'avg_write_size': 'avg_write_KB',
            'max_read_size': 'max_read_KB',
            'max_write_size': 'max_write_KB',
            'read_count': 'read_ops',
            'write_count': 'write_ops'
        }
        for old, new in mapping.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old] / (1024 if 'size' in old else 1)

        records = []
        for _, row in df.iterrows():
            opt, pred_tp = self.find_optimal_chunk_size(
                self.model, row, self.feature_names, self.candidate_chunks
            )
            records.append({
                'file_path': row.get('file_path', ''),
                'file_size_KB': row.get('file_size_KB', np.nan),
                'current_chunk_KB': row['chunk_size_KB'],
                'optimal_chunk_KB': opt,
                'predicted_throughput': pred_tp
            })

        for rec in records:
            print(f"File: {rec['file_size_KB']}, Current: {rec['current_chunk_KB']} KB, "
                  f"Optimal: {rec['optimal_chunk_KB']} KB, "
                  f"Predicted TP: {rec['predicted_throughput']:.2f} KB/s")

        df_preds = pd.DataFrame(records)
        best_idx = df_preds['predicted_throughput'].idxmax()
        return int(df_preds.loc[best_idx, 'optimal_chunk_KB'])
