from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
)
import numpy as np
import torch


def compute_metrics(eval_pred, label2id):
    metrics = {}
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    # Threshold at 0.5 for multilabel binary
    preds = (probs > 0.5).astype(int)

    batch_size, seq_len, n_labels = preds.shape
    batch_size, seq_len, n_labels = labels.shape

    preds = preds.reshape(batch_size * seq_len, n_labels)
    labels = labels.reshape(batch_size * seq_len, n_labels)

    for label, id in label2id.items():
        class_report = classification_report(
            labels[:, id].tolist(),
            preds[:, id].tolist(),
            zero_division=0,
            output_dict=True,
            labels=[0, 1],
            target_names=[f"not_{label}", label]
        )
        metrics.update(
            {
                f"f1-{label}": class_report[label]["f1-score"],
                f"precision-{label}": class_report[label]["precision"],
                f"recall-{label}": class_report[label]["recall"],
                f"kappa-{label}": cohen_kappa_score(labels[:, id], preds[:, id]),
            }
        )

    # metrics.update(
    #     {
    #         "accuracy": accuracy_score(labels, preds),
    #         "f1-micro": f1_score(labels, preds, average="micro", zero_division=0),
    #         "f1-macro": f1_score(labels, preds, average="macro", zero_division=0),
    #         "precision-micro": precision_score(
    #             labels, preds, average="micro", zero_division=0
    #         ),
    #         "precision-macro": precision_score(
    #             labels, preds, average="macro", zero_division=0
    #         ),
    #         "recall-micro": recall_score(
    #             labels, preds, average="micro", zero_division=0
    #         ),
    #         "recall-macro": recall_score(
    #             labels, preds, average="macro", zero_division=0
    #         ),
    #     }
    # )

    return metrics
