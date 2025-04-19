import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(output_path):
    """
    Evaluate the model's predictions against the actual values.

    Args:
        output_path (str): Path to the output CSV file containing actual and predicted values.
    """
    # Load the output data
    df = pd.read_csv(output_path)

    # Check if required columns exist
    if 'OT' not in df.columns or 'predicted_OT' not in df.columns:
        logger.error("Output CSV must contain 'OT' and 'predicted_OT' columns.")
        return

    # Extract actual and predicted values
    y_true = df['OT']
    y_pred = df['predicted_OT']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Log the results
    logger.info("\nEvaluation Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    # Specify the path to the output CSV file
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "output.csv"
    # output_csv_path = "XGBoost/output.csv"  # Change this to your actual output file path
    evaluate_model(data_path)