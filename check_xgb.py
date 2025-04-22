import pandas as pd
from beechunker.ml.xgboost_model import BeeChunkerXGBoost


def configure_display():
    """Set Pandas options to display the full DataFrame without truncation."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)


def reset_display():
    """Reset Pandas display options to default."""
    pd.reset_option("all")


def main():
    """Train the BeeChunkerXGBoost model on the training dataset."""
    xgb = BeeChunkerXGBoost()
    df = pd.read_csv("/home/jgajbha/BeeChunker/test_results28k_filtered.csv")

    configure_display()

    result = xgb.train(df)
    print("\n📊 Training Result:\n")
    print(result)

    reset_display()


def predict():
    """Use the BeeChunkerXGBoost model to make predictions on the test dataset."""
    xgb = BeeChunkerXGBoost()
    df = pd.read_csv("/home/jgajbha/BeeChunker/test.csv")

    configure_display()

    res = xgb.predict(df)
    print("\n🔮 Prediction Results:\n")
    print(res)


if __name__ == "__main__":
    main()
    predict()