from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
)
import numpy as np


def compute_metrics(eval_pred, label2id):
    metrics = {}
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    # Threshold at 0.5 for multilabel binary
    preds = (probs > 0.5).astype(int)
    
    for label, id in label2id.items():
        class_names = [f"not_{label}", label]
        class_report = classification_report(
            labels[:, id].tolist(),
            preds[:, id].tolist(),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        metrics.update(
            {
                f"f1_{label}": class_report[label]["f1-score"],
                f"precision_{label}": class_report[label]["precision"],
                f"recall_{label}": class_report[label]["recall"],
                f"kappa_{label}": cohen_kappa_score(labels[:, id], preds[:, id]),
            }
        )

    metrics.update(
        {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_micro": precision_score(
                labels, preds, average="micro", zero_division=0
            ),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                labels, preds, average="micro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }
    )

    return metrics