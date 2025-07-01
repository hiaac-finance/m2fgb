import numpy as np
from scipy.sparse import csr_matrix


def projection_to_simplex(mu: np.ndarray, z: float = 1) -> np.ndarray:
    """Project the vector mu to the closest vector with L1 norm equal to z."""
    sorted_mu = mu[np.argsort(mu)]
    n = len(mu)
    t = np.mean(mu) - z / n
    for i in range(len(mu) - 2, -1, -1):
        t_i = np.mean(sorted_mu[(i + 1) :]) - z / (n - i - 1)
        if t_i >= sorted_mu[i]:
            t = t_i
            break

    x = mu - t
    x = np.where(x > 0, x, 0)
    return x


def logloss_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute the gradient for cross entropy log loss."""
    grad = -(y_true - y_pred)
    return grad


def logloss_hessian(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute the hessian for cross entropy log loss."""
    hess = y_pred * (1 - y_pred)
    return hess


def logloss_group(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subgroup: np.ndarray,
    fairness_constraint: str,
) -> np.ndarray:
    """For each subgroup, calculates the mean log loss of the samples.


    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities of the positive class.
    y_true : np.ndarray
        True labels of the samples.
    subgroup : np.ndarray
        Subgroup indicator for each sample.
    fairness_constraint : str
        Fairness constraint to apply. Options are:
        - "equalized_loss": Equalized loss across subgroups.
        - "positive_rate": Positive rate across subgroups.
        - "true_positive_rate": True positive rate across subgroups.
        - "true_negative_rate": True negative rate across subgroups.

    Returns
    -------
    np.ndarray
        Mean log loss for each subgroup.

    Raises
    ------
    ValueError
        If the fairness constraint is not supported.
    """
    if fairness_constraint == "equalized_loss":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if fairness_constraint == "positive_rate":
        y_ = np.ones(y_true.shape[0])  # all positive class
        loss = -(y_ * np.log(y_pred) + (1 - y_) * np.log(1 - y_pred))
    elif fairness_constraint == "true_positive_rate":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss[y_true == 0] = np.nan  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss[y_true == 1] = np.nan  # only consider the loss of the positive class
    else:
        raise ValueError(f"Fairness constraint {fairness_constraint} is not supported.")

    # smart numpy groupby that assumes that subgroup is sorted
    loss = np.column_stack((loss, subgroup))
    loss = np.split(loss[:, 0], np.unique(loss[:, 1], return_index=True)[1][1:])
    loss = np.array([np.nanmean(l) for l in loss])
    return loss


def logloss_group_grad(
    y_pred: np.ndarray, y_true: np.ndarray, fairness_constraint: str
) -> np.ndarray:
    """Create an array with the gradient of fairness metrics."""
    if fairness_constraint == "equalized_loss":
        grad = -(y_true - y_pred)
    elif fairness_constraint == "positive_rate":
        y_ = np.ones(y_true.shape[0])  # all positive class
        grad = -(y_ - y_pred)
    elif fairness_constraint == "true_positive_rate":
        grad = -(y_true - y_pred)
        grad[y_true == 0] = 0  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        grad = -(y_true - y_pred)
        grad[y_true == 1] = 0  # only consider the loss of the negative class
    else:
        raise ValueError(f"Fairness constraint {fairness_constraint} is not supported.")

    return grad


def logloss_group_hess(
    y_pred: np.ndarray, y_true: np.ndarray, fairness_constraint: str
) -> np.ndarray:
    """Create an array with the hessian of fairness metrics."""
    if (
        fairness_constraint == "equalized_loss"
        or fairness_constraint == "positive_rate"
    ):
        hess = y_pred * (1 - y_pred)
    elif fairness_constraint == "true_positive_rate":
        hess = y_pred * (1 - y_pred)
        hess[y_true == 0] = 0  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        hess = y_pred * (1 - y_pred)
        hess[y_true == 1] = 0
    else:
        raise ValueError(f"Fairness constraint {fairness_constraint} is not supported.")

    return hess


def get_subgroup_indicator(subgroup: np.ndarray) -> csr_matrix:
    groups = np.unique(subgroup)
    n = len(subgroup)
    I = np.zeros((subgroup.shape[0], len(groups)))
    n_g_max = -np.inf
    for i, g in enumerate(groups):
        n_g = np.sum(subgroup == g)
        n_g_max = max(n_g_max, n_g)
        I[subgroup == g, i] = 1 / np.sum(subgroup == g)

    I = I * n
    I = csr_matrix(I)
    return I


def max_logloss_score(
    y_ground: np.ndarray,
    y_prob: np.ndarray,
    A: np.ndarray,
    fairness_constraint: str = "equalized_loss",
) -> float:
    """Calculate the minimum mean loss of the groups. The loss is binary cross entropy.
    It work with multiple groups.

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted probabilities of the positive class
    A : ndarray
        Group labels

    Returns
    -------
    float
        Minimum mean loss of groups
    """
    logloss = logloss_group(y_ground, y_prob, A, fairness_constraint)
    return max(logloss)


def squaredloss_group(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subgroup: np.ndarray,
    fairness_constraint: str = "",
):
    """For each subgroup, calculates the mean log loss of the samples."""
    loss = (y_true - y_pred) ** 2

    # smart numpy groupby that assumes that subgroup is sorted
    loss = np.column_stack((loss, subgroup))
    loss = np.split(loss[:, 0], np.unique(loss[:, 1], return_index=True)[1][1:])
    loss = np.array([np.mean(l) for l in loss])
    return loss


def squaredloss_grad(
    y_pred: np.ndarray, y_true: np.ndarray, fairness_constraint: str = ""
):
    """Calculate the gradient of the squared loss."""
    grad = 2 * (y_pred - y_true)
    return grad


def squaredloss_hess(
    y_pred: np.ndarray, y_true: np.ndarray, fairness_constraint: str = ""
):
    """Calculate the hessian of the squared loss."""
    hess = 2 * np.ones(y_pred.shape[0])
    return hess


def max_mse(y_ground: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    """Calculate the worst group MSE.

    Parameters
    ----------
    y_ground : np.ndarray
        Real ground truth values.
    y_pred : np.ndarray
        Predicted values.
    A : np.ndarray
        Group labels.

    Returns
    -------
    float
        Worst group MSE.
    """
    max_mse = -np.inf
    for a in np.unique(A):
        mse = np.mean((y_ground[A == a] - y_pred[A == a]) ** 2)
        max_mse = max(max_mse, float(mse))
    return max_mse
