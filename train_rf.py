import os
import pandas as pd
from beechunker.ml.random_forest import BeeChunkerRF
from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer

# Paths
TRAIN_CSV = "/home/agupta72/Chunker/test_results28k.csv"
TEST_CSV  = "/home/agupta72/Chunker/test.csv"

def main():
    # Initialize the RF trainer/predictor
    rf = BeeChunkerRF()

    # Always retrain the model (will overwrite any existing model files)
    print(f"Training model on: {TRAIN_CSV}")
    success = rf.train(TRAIN_CSV)
    if not success:
        print("Error: Model training failed, aborting.")
        return
    print("Model trained and saved successfully.")

    # Load test inputs
    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded test data, shape={df_test.shape}")

    # Run prediction
    preds = rf.predict(df_test)
    if preds is None:
        print("Error: Prediction failed.")
        return
    print("Sample predictions:\n", preds)

    # Example: print first optimal chunk
    opt_ck = int(preds.iloc[0]['optimal_chunk_KB'])
    print(f"Optimal chunk size for first file: {opt_ck} KB")


def test_optimizer():
    optimizer = ChunkSizeOptimizer()
    file_path = "/mnt/beegfs/dummy_file.txt"

    # Get current chunk size
    curr_ck = optimizer.get_current_chunk_size(file_path)
    if curr_ck is None:
        print(f"Cannot determine current chunk size for {file_path}")
        return
    print(f"Current chunk size: {curr_ck} KB")

    # Optimize file chunk size using RF model
    result = optimizer.optimize_file(file_path, model_type="rf")
    print("Optimizer result:\n", result)


if __name__ == "__main__":
    main()
    # test_optimizer()  # uncomment to run optimizer integration check
