import pandas as pd
from beechunker.ml.random_forest import BeeChunkerRF

def test_manual_df_predict():
    # ── Manually define your test DataFrame here ───────────────
    data = [
        {
            'file_path': 'fileA',
            'file_size_KB': 204800,
            'chunk_size_KB': 1733,
            'access_count': 20,
            'avg_read_KB': 57.7,
            'avg_write_KB': 70.3,
            'max_read_KB': 130,
            'max_write_KB': 124,
            'read_ops': 12,
            'write_ops': 8,
            'throughput_KBps': 25624.04,
            'error_message': ''
        }
    ]
    df_input = pd.DataFrame(data)
    print("Input DataFrame:")
    print(df_input)

    # Initialize and load the model
    predictor = BeeChunkerRF()
    if not predictor.load():
        raise RuntimeError("Failed to load trained RF model. Please train the model first.")
    print("Model loaded successfully.")

    # Run prediction
    optimal_chunk = predictor.predict(df_input)
    print("\nOptimal chunk size recommendation (KB):", optimal_chunk)


if __name__ == '__main__':
    test_manual_df_predict()





# import os
# import pandas as pd
# import joblib

# from beechunker.ml.feature_extraction import OptimalThroughputProcessor
# from beechunker.ml.random_forest import BeeChunkerRF

# def main():
#     # Paths
#     raw_csv = "/home/agupta72/Chunker/test_results28k_filtered.csv"
#     models_dir = os.getenv("BEECHUNKER_MODELS_DIR", "/home/agupta72/Chunker/")

#     # 1) Check raw data exists
#     assert os.path.exists(raw_csv), f"Raw CSV not found at {raw_csv}"

#     # 2) OT processing test
#     processed_csv = os.path.join(models_dir, "data/test_processed.csv")
#     proc = OptimalThroughputProcessor(raw_csv, processed_csv, quantile=0.65)
#     proc.run()
#     df_proc = pd.read_csv(processed_csv)
#     print(f"Processed OT data shape: {df_proc.shape}")
#     assert "OT" in df_proc.columns, "OT column missing after processing"

#     # 3) Training test
#     trainer = BeeChunkerRF()
#     success = trainer.train(raw_csv)
#     assert success, "Model training failed"

#     # 4) Check model artifacts
#     expected_files = [
#         "rf_model.joblib",
#         "rf_base.joblib",
#         "hgb_base.joblib",
#         "logistic_meta.joblib",
#         "last_training_time.txt"
#     ]
#     for fname in expected_files:
#         path = os.path.join(models_dir, f"models/{fname}")
#         assert os.path.exists(path), f"Expected artifact missing: {fname}"
#     print("All model artifacts present.")

#     # 5) Loading test
#     predictor = BeeChunkerRF()
#     assert predictor.load(), "Loading saved model failed"

#     # 6) Prediction test
#     raw_df = pd.read_csv(raw_csv)
#     sample_df = raw_df.head(5)
#     preds = predictor.predict(sample_df)
#     print("Sample predictions:\n", preds)
#     assert not preds.empty, "No predictions returned"
#     required_cols = {"file_path","file_size_KB","current_chunk_KB","optimal_chunk_KB","confidance"}
#     missing = required_cols - set(preds.columns)
#     assert not missing, f"Missing prediction columns: {missing}"
#     print("Prediction test passed.")


# if __name__ == "__main__":
#     main()
#     print("Running RF test script...")




# import os
# import pandas as pd

# from beechunker.ml.random_forest import BeeChunkerRF
# from beechunker.ml.feature_extraction import OptimalThroughputProcessor

# # Configuration
# default_raw_csv = "/home/agupta72/Chunker/test_new.csv"
# default_models_dir = os.getenv("BEECHUNKER_MODELS_DIR", "/home/agupta72/Chunker/models")


# def main(raw_csv=default_raw_csv, models_dir=default_models_dir):
#     # 1) Ensure raw input exists
#     if not os.path.exists(raw_csv):
#         raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")

#     # 2) Load and preprocess OT labels (optional, if you want to inspect)
#     processed_csv = os.path.join(models_dir, "test_processed.csv")
#     proc = OptimalThroughputProcessor(raw_csv, processed_csv, quantile=0.65)
#     proc.run()
#     df_proc = pd.read_csv(processed_csv)
#     print(f"Processed OT data shape: {df_proc.shape}")

#     # 3) Load the trained model
#     predictor = BeeChunkerRF()
#     if not predictor.load():
#         raise RuntimeError("Failed to load trained model from " + models_dir)
#     print("Model loaded successfully from {}".format(models_dir))

#     # 4) Read input features for prediction
#     df_input = pd.read_csv(raw_csv)
#     print(f"Input data shape: {df_input.shape}")

#     # 5) Predict optimal chunk sizes
#     predictions = predictor.predict(df_input)
#     if predictions is None:
#         raise RuntimeError("Prediction returned None. Check model and input preprocessing.")

#     # 6) Display results
#     print("Sample predictions:")
#     print(predictions.head())

#     # 7) Example: optimal chunk for first row
#     first_opt = int(predictions.iloc[0]['optimal_chunk_KB'])
#     first_prob = float(predictions.iloc[0].get('confidence', predictions.iloc[0].get('confidance', None)))
#     print(f"Optimal chunk size for first file: {first_opt} KB (confidence {first_prob:.4f})")


# if __name__ == '__main__':
#     main()




# import os
# import pandas as pd

# from beechunker.ml.random_forest import BeeChunkerRF
# from beechunker.ml.feature_extraction import OptimalThroughputProcessor

# # Configuration
# default_raw_csv = "/home/agupta72/Chunker/test_results28k_filtered.csv"

# default_models_dir = os.getenv("BEECHUNKER_MODELS_DIR", "/home/agupta72/Chunker/models")


# def cleanup_models(models_dir):
#     """
#     Remove existing model artifacts in models_dir.
#     """
#     artifacts = [
#         "rf_model.joblib",
#         "rf_base.joblib",
#         "hgb_base.joblib",
#         "logistic_meta.joblib",
#         "feature_names.joblib",
#         "candidate_chunks.joblib",
#         "last_training_time.txt"
#     ]
#     for fname in artifacts:
#         path = os.path.join(models_dir, fname)
#         if os.path.exists(path):
#             os.remove(path)
#             print(f"Deleted existing artifact: {fname}")


# def main(raw_csv=default_raw_csv, models_dir=default_models_dir):
#     # 1) Ensure raw input exists
#     if not os.path.exists(raw_csv):
#         raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")

#     # 2) Preprocess OT labels (optional)
#     processed_csv = os.path.join(models_dir, "temp_processed.csv")
#     proc = OptimalThroughputProcessor(raw_csv, processed_csv, quantile=0.65)
#     proc.run()
#     df_proc = pd.read_csv(processed_csv)
#     print(f"Processed OT data shape: {df_proc.shape}")
#     os.remove(processed_csv)

#     # 3) Initialize model handler
#     predictor = BeeChunkerRF()

#     # 4) Check for existing model and retrain if present
#     model_file = os.path.join(predictor.models_dir, "rf_model.joblib")
#     if os.path.exists(model_file):
#         print("Existing model detected. Removing old artifacts and retraining...")
#         cleanup_models(predictor.models_dir)
#         trained = predictor.train(raw_csv)
#         if not trained:
#             raise RuntimeError("Retraining failed.")
#         print("Retraining complete.")
#     else:
#         print("No existing model found. Training new model...")
#         trained = predictor.train(raw_csv)
#         if not trained:
#             raise RuntimeError("Initial training failed.")
#         print("Model training complete.")

#     # 5) Read input features for prediction
#     df_input = pd.read_csv(raw_csv)
#     print(f"Input data shape: {df_input.shape}")

#     # 6) Predict optimal chunk sizes
#     predictions = predictor.predict(df_input)
#     if predictions is None:
#         raise RuntimeError("Prediction failed.")

#     # 7) Display results
#     print("Sample predictions:")
#     print(predictions.head())

#     # 8) Example: optimal chunk for first row
#     first = predictions.iloc[0]
#     opt_ck = int(first['optimal_chunk_KB'])
#     conf = first.get('confidence', first.get('confidance', None))
#     print(f"Optimal chunk size for first file: {opt_ck} KB (confidence {conf:.4f})")


# if __name__ == '__main__':
#     main()
