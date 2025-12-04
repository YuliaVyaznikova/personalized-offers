import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

def metric_report(y_true, y_pred, ks=(1, 3, 5, 10, 20), output_dir="metrics_output"):
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]

    n = len(y_true)
    total_pos = y_true.sum()

    precision_at_k = {}
    recall_at_k = {}

    for k in ks:
        if k <= 1:
            top_k = k
        else:
            top_k = int(n * (k / 100))

        top_k = max(1, top_k)
        y_top = y_true_sorted[:top_k]

        if k <= 10:
            precision_at_k[str(k)] = y_top.mean()

        if k <= 20:
            recall_at_k[str(k)] = y_top.sum() / total_pos if total_pos > 0 else 0

    
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
    plt.close()

  
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=200)
    plt.close()

    
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision@k": precision_at_k,
        "recall@k": recall_at_k
    }

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4))

    return metrics
