import os
import pandas as pd
import joblib

from beechunker.ml.feature_extraction import OptimalThroughputProcessor
from beechunker.ml.random_forest import BeeChunkerRF

def main():
    # Paths
    raw_csv = "/home/agupta72/Chunker/test_new.csv"
    models_dir = os.getenv("BEECHUNKER_MODELS_DIR", "/home/agupta72/Chunker/")

    # 1) Check raw data exists
    assert os.path.exists(raw_csv), f"Raw CSV not found at {raw_csv}"

    # 2) OT processing test
    processed_csv = os.path.join(models_dir, "data/test_processed.csv")
    proc = OptimalThroughputProcessor(raw_csv, processed_csv, quantile=0.65)
    proc.run()
    df_proc = pd.read_csv(processed_csv)
    print(f"Processed OT data shape: {df_proc.shape}")
    assert "OT" in df_proc.columns, "OT column missing after processing"

    # 3) Training test
    trainer = BeeChunkerRF()
    success = trainer.train(raw_csv)
    assert success, "Model training failed"

    # 4) Check model artifacts
    expected_files = [
        "rf_model.joblib",
        "rf_base.joblib",
        "hgb_base.joblib",
        "logistic_meta.joblib",
        "last_training_time.txt"
    ]
    for fname in expected_files:
        path = os.path.join(models_dir, f"models/{fname}")
        assert os.path.exists(path), f"Expected artifact missing: {fname}"
    print("All model artifacts present.")

    # 5) Loading test
    predictor = BeeChunkerRF()
    assert predictor.load(), "Loading saved model failed"

    # 6) Prediction test
    raw_df = pd.read_csv(raw_csv)
    sample_df = raw_df.head(5)
    preds = predictor.predict(sample_df)
    print("Sample predictions:\n", preds)
    assert not preds.empty, "No predictions returned"
    required_cols = {"file_path","file_size_KB","current_chunk_KB","optimal_chunk_KB","confidance"}
    missing = required_cols - set(preds.columns)
    assert not missing, f"Missing prediction columns: {missing}"
    print("Prediction test passed.")


if __name__ == "__main__":
    main()
    print("Running RF test script...")