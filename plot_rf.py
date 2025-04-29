# data_analysis.py

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────
INPUT_CSV   = "/home/agupta72/Chunker/test_results28k_filtered.csv"
OUTPUT_DIR  = "."                 # where to save stats + plots
TARGET_MB   = 1000                  # the file‐size (in MB) you want to hold constant
TOL_MB      = 1                   # tolerance around TARGET_MB, in MB
# ─────────────────────────────────────────────────────────────────

def size_bucket(mb):
    if mb <= 10:
        return "Small (≤10 MB)"
    elif mb <= 500:
        return "Medium (10–500 MB)"
    else:
        return "Large (>500 MB)"

def main():
    # 1) Load
    df = pd.read_csv(INPUT_CSV)
    df["file_size_MB"] = df["file_size_KB"] / 1024

    # 2) Buckets
    df["size_group"] = df["file_size_MB"].apply(size_bucket)

    # 3) Group‐level stats
    stats = df.groupby("size_group")["throughput_KBps"] \
        .agg(mean="mean", std="std", var="var") \
        .reset_index()
    print("\n=== Throughput by Size Group ===")
    print(stats.to_string(index=False))
    stats.to_csv(os.path.join(OUTPUT_DIR, "group_stats.csv"), index=False)

    # 4) Bar chart of mean ± std
    plt.figure()
    x = stats["size_group"]
    y = stats["mean"]
    err = stats["std"]
    plt.bar(x, y, yerr=err, capsize=5)
    plt.xlabel("File‐Size Group")
    plt.ylabel("Mean Throughput (KB/s)")
    plt.title("Mean ± Std Dev of Throughput by File‐Size Group")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "throughput_by_size_group.png"))
    print("→ Saved throughput_by_size_group.png")

    # 5) Throughput vs Chunk‐Size at a fixed file size
    tgt_kb   = TARGET_MB * 1024
    tol_kb   = TOL_MB    * 1024
    sub = df[np.abs(df["file_size_KB"] - tgt_kb) <= tol_kb]
    if sub.empty:
        print(f"\nNo rows found within ±{TOL_MB} MB of {TARGET_MB} MB.")
        return

    chunk_stats = sub.groupby("chunk_size_KB")["throughput_KBps"] \
                     .mean().reset_index()
    best = chunk_stats.loc[chunk_stats["throughput_KBps"].idxmax()]

    plt.figure()
    plt.plot(chunk_stats["chunk_size_KB"], chunk_stats["throughput_KBps"], marker="o")
    plt.xlabel("Chunk Size (KB)")
    plt.ylabel("Mean Throughput (KB/s)")
    plt.title(f"Throughput vs Chunk Size\n(at ≈{TARGET_MB} MB ±{TOL_MB} MB)")

    # annotate each point with its chunk_size value
    for _, row in chunk_stats.iterrows():
        x = row["chunk_size_KB"]
        y = row["throughput_KBps"]
        plt.annotate(
            str(int(x)),
            xy=(x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    plt.xlim(0, 66000)             # limit x-axis
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "throughput_vs_chunk.png"))
    print("→ Saved throughput_vs_chunk.png")

    print(f"\nOptimal chunk for ~{TARGET_MB} MB is "
          f"{int(best['chunk_size_KB'])} KB "
          f"(mean throughput = {best['throughput_KBps']:.1f} KB/s)")

    # 6) Boxplot of throughput by size group
    plt.figure()
    plt.boxplot(
        [df[df["size_group"] == grp]["throughput_KBps"] for grp in stats["size_group"]],
        labels=stats["size_group"],
        showmeans=True,
        meanline=True,
        meanprops={"color": "red", "linestyle": "--"}
    )
    plt.xlabel("File‐Size Group")
    plt.ylabel("Throughput (KB/s)")
    plt.title("Throughput by File‐Size Group")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_throughput_by_size.png"))
    print("→ Saved boxplot_throughput_by_size.png")
    print("\n=== Boxplot of Throughput by Size Group ===")
    print(df.groupby("size_group")["throughput_KBps"].describe().to_string())
    print("\n→ Boxplot saved as boxplot_throughput_by_size.png")

    # ─── 7) Per–File‐Size Statistics ─────────────────────────────────────────────
    # Group by each unique file_size_KB
    size_stats = df.groupby("file_size_KB")["throughput_KBps"] \
                   .agg(mean="mean", std="std", var="var") \
                   .reset_index() \
                   .sort_values("file_size_KB")

    # Save to CSV and print top/bottom few rows
    size_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_per_file_size.csv"), index=False)
    print("\n=== Stats per unique file_size_KB ===")
    print(size_stats.head(10).to_string(index=False))
    print("...\n")
    print(size_stats.tail(10).to_string(index=False))

    # ─── Export per‐file‐size stats as table ────────────────────────────────────
    size_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_per_file_size.csv"), index=False)
    size_stats.to_excel(os.path.join(OUTPUT_DIR, "stats_per_file_size.xlsx"), index=False)
    print("→ Exported stats_per_file_size.csv and stats_per_file_size.xlsx")

    # Optional: Plot mean throughput vs. file size (in MB)
    plt.figure(figsize=(8,4))
    plt.plot(
        size_stats["file_size_KB"] / 1024, 
        size_stats["mean"], 
        marker=".", 
        linestyle="none", 
        alpha=0.6
    )
    plt.xlabel("File Size (MB)")
    plt.ylabel("Mean Throughput (KB/s)")
    plt.title("Mean Throughput by Exact File Size")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_throughput_per_file_size.png"))
    print("→ Saved mean_throughput_per_file_size.png")
    
    # 8) Correlation matrix (numeric features only)
    num_df = df.select_dtypes(include=["number"])
    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
    print("→ Saved correlation_matrix.png")

    print("\n=== Correlation Matrix ===")
    print(corr.to_string())

if __name__ == "__main__":
    main()

    # # 5) Throughput vs Chunk‐Size at a fixed file size
    # tgt_kb   = TARGET_MB * 1024
    # tol_kb   = TOL_MB    * 1024
    # sub = df[np.abs(df["file_size_KB"] - tgt_kb) <= tol_kb]
    # if sub.empty:
    #     print(f"\nNo rows found within ±{TOL_MB} MB of {TARGET_MB} MB.")
    #     return

    # chunk_stats = sub.groupby("chunk_size_KB")["throughput_KBps"] \
    #                  .mean().reset_index()
    # best = chunk_stats.loc[chunk_stats["throughput_KBps"].idxmax()]

    # plt.figure()
    # plt.plot(chunk_stats["chunk_size_KB"], chunk_stats["throughput_KBps"], marker="o")
    # plt.xlabel("Chunk Size (KB)")
    # plt.ylabel("Mean Throughput (KB/s)")
    # plt.title(f"Throughput vs Chunk Size\n(at ≈{TARGET_MB} MB ±{TOL_MB} MB)")
    # plt.xlim(0, 10000)             # <- limit x-axis to [0, 10000]
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "throughput_vs_chunk.png"))
    # print("→ Saved throughput_vs_chunk.png")

    # print(f"\nOptimal chunk for ~{TARGET_MB} MB is "
    #       f"{int(best['chunk_size_KB'])} KB "
    #       f"(mean throughput = {best['throughput_KBps']:.1f} KB/s)")


# # test_beechunker_rf.py

# import os
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_curve,
#     auc,
#     precision_recall_curve,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )
# from beechunker.ml.random_forest import BeeChunkerRF
# from beechunker.ml.feature_extraction import OptimalThroughputProcessor
# from beechunker.common.config import config

# # ─── CONFIGURE THESE ──────────────────────────────────────────────────────────
# MODEL_DIR = "/home/agupta72/Chunker/models"
# RAW_CSV   = "/home/agupta72/Chunker/test_results28k_filtered.csv"   
# # ───────────────────────────────────────────────────────────────────────────────

# def main():
#     raw_path = os.path.join(MODEL_DIR, RAW_CSV)
#     assert os.path.exists(raw_path), f"Raw CSV not found: {raw_path}"
#     print(f"▶ Raw data: {raw_path}")

#     # 1) Run the OT‐label preprocessing
#     tmp_out = os.path.join(MODEL_DIR, "_tmp_test_proc.csv")
#     proc = OptimalThroughputProcessor(
#         input_csv=raw_path,
#         output_csv=tmp_out,
#         quantile=config.get("ml", "ot_quantile")
#     )
#     proc.run()

#     # 2) Load the processed file (now has an 'OT' column)
#     df = pd.read_csv(tmp_out)
#     os.remove(tmp_out)
#     print("▶ After processing, columns are:", df.columns.tolist())

#     # 3) Numeric subset and drop OT + throughput
#     num = df.select_dtypes(include="number")
#     # auto‐detect label + throughput names
#     label_col = "OT" if "OT" in num.columns else "ot"
#     thr_col   = "throughput_KBps" if "throughput_KBps" in num.columns else "throughput_kbps"
#     X = num.drop(columns=[label_col, thr_col])
#     y = num[label_col]
#     print(f"▶ Features for ML: {X.columns.tolist()}")
#     print(f"▶ Label = {label_col}, dropping {thr_col}")

#     # 4) Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size = config.get("ml", "test_size"),
#         stratify = y,
#         random_state = 42
#     )

#     # 5) Train (will write into MODEL_DIR)
#     bc = BeeChunkerRF()
#     if not bc.train(df):
#         raise RuntimeError("Training failed!")
#     print("▶ Training succeeded; artifacts in", bc.models_dir)

#     # 6) Load back models & feature list
#     stack      = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
#     rf_base    = joblib.load(os.path.join(MODEL_DIR, "rf_base.joblib"))
#     feat_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

#     # 7) Predict & score
#     y_pred  = stack.predict(X_test)
#     y_score = stack.predict_proba(X_test)[:, 1]

#     # ─── A. ROC Curve ───────────────────────────────────────────────────────────
#     fpr, tpr, _ = roc_curve(y_test, y_score)
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
#     plt.plot([0,1],[0,1],"--", lw=1)
#     plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.legend(loc="best")
#     plt.savefig("roc_curve.png")
#     print("→ Saved roc_curve.png")

#     # ─── B. Precision–Recall Curve ───────────────────────────────────────────────
#     precision, recall, _ = precision_recall_curve(y_test, y_score)
#     pr_auc = auc(recall, precision)
#     plt.figure()
#     plt.plot(recall, precision, lw=2, label=f"AUC = {pr_auc:.3f}")
#     plt.title("Precision–Recall"); plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.legend(loc="best")
#     plt.savefig("pr_curve.png")
#     print("→ Saved pr_curve.png")

#     # ─── C. Confusion Matrix ─────────────────────────────────────────────────────
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.savefig("confusion_matrix.png")
#     print("→ Saved confusion_matrix.png")

#     # ─── D. RF Feature Importances ───────────────────────────────────────────────
#     importances = rf_base.feature_importances_
#     idx = importances.argsort()[::-1]
#     names = [feat_names[i] for i in idx]
#     plt.figure(figsize=(8,6))
#     plt.bar(range(len(importances)), importances[idx])
#     plt.xticks(range(len(importances)), names, rotation=90)
#     plt.title("RF Feature Importances"); plt.tight_layout()
#     plt.savefig("feature_importances.png")
#     print("→ Saved feature_importances.png")

# if __name__ == "__main__":
#     main()