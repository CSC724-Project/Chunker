#!/usr/bin/env python3
import pandas as pd
from beechunker.ml.temp_xgboost import BeeChunkerXGB

# === User‑defined file paths ===
TRAIN_INPUT_PATH = "/home/agupta72/Chunker/test_results28k_filtered.csv"
PRED_INPUT_PATH  = "/home/agupta72/Chunker/test.csv"


def train_model(train_csv: str):
    """
    Train the BeeChunkerXGB model once and persist it to disk.
    """
    model = BeeChunkerXGB()
    success = model.train(train_csv)
    if not success:
        raise RuntimeError(f"Training failed on {train_csv}")
    print(f"✅ Training completed on {train_csv}")


def predict_optimal_chunk(pred_csv: str) -> int:
    """
    Load the persisted model, run predict() on new data,
    and return (and print) the single optimal chunk size in KB.
    """
    model = BeeChunkerXGB()
    # read new raw data
    df = pd.read_csv(pred_csv)
    # predict returns an int
    optimal_chunk = model.predict(df)
    print(optimal_chunk)
    return optimal_chunk


if __name__ == "__main__":
    # === Uncomment the following line to train the model (do this only once) ===
    # train_model(TRAIN_INPUT_PATH)

    # === Then, predict whenever you have new data ===
    predict_optimal_chunk(PRED_INPUT_PATH)