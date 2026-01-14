# src/eval/metrics.py

from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return confusion_matrix(y_true, y_pred)
