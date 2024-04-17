# Necessary packages
import numpy as np
from sklearn.metrics import accuracy_score

__all__ = ["rmse_error", "reidentify_score"]


def rmse_error(y_true, y_pred):
    """User defined root mean squared error.

    Args:
    - y_true: true labels
    - y_pred: predictions

    Returns:
    - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred) ** 2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse


def reidentify_score(enlarge_label, pred_label):
    """Return the reidentification score.

    Args:
    - enlarge_label: 1 for train data, 0 for other data
    - pred_label: 1 for reidentified data, 0 for not reidentified data

    Returns:
    - accuracy: reidentification score
    """
    accuracy = accuracy_score(enlarge_label, pred_label > 0.5)
    return accuracy


if __name__ == "__main__":
    pass
