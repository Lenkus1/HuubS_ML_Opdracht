from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV

from src import protocol


def class_balance(x: np.ndarray, col: str) -> None:
    pd.DataFrame(x, columns=[col]).groupby(col).size().plot.bar()


def cfm_heatmap(
    cfm: np.ndarray,
    figsize: tuple = (8, 8),
    scale: float = None,
    labels: List[str] = None,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    figsize: tuple, default (8,8)
    scale: string. The direction over which the numbers are scaled.
        Either None, 'total', 'rowwise' or 'colwise'
    """

    if scale == "total":
        cfm_norm = cfm / np.sum(cfm)
    elif scale == "rowwise":
        cfm_norm = cfm / np.sum(cfm, axis=1, keepdims=True)
    elif scale == "colwise":
        cfm_norm = cfm / np.sum(cfm, axis=0, keepdims=True)
    else:
        cfm_norm = cfm
    plt.figure(figsize=figsize)
    if labels is not None:
        plot = sns.heatmap(
            cfm_norm,
            annot=cfm_norm,
            vmin=vmin,
            vmax=vmax,
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        plot = sns.heatmap(cfm_norm, annot=cfm_norm, vmin=vmin, vmax=vmax)
    plot.set(xlabel="Predicted", ylabel="Target")


def precision_recall_curve(
    precision: np.ndarray, recall: np.ndarray, threshold: np.ndarray
) -> None:
    """Takes as input arrays with precision-recall values and threshold
    checks if precision and recall are a long as the thresholds (somehow, the
    output of sklearn precision_recall_curve does not has similar length) and
    cuts of the part of p/r that are too long

    Args:
        precision (np.ndarray): [description]
        recall (np.ndarray): [description]
        threshold (np.ndarray): [description]
    """

    n = len(threshold)
    if (n != len(precision)) or (n != len(recall)):
        logger.info(
            f"Found different lengths: thresholds={n} precision={len(precision)} \
                and recall={len(recall)}",
        )
        logger.info(f"Stripping of items after index {n}")
        precision = precision[:n]
        recall = recall[:n]

    data = pd.DataFrame(
        {"precision": precision, "recall": recall, "threshold": threshold}
    )
    sns.lineplot(x="threshold", y="precision", label="precision", data=data)
    sns.lineplot(x="threshold", y="recall", label="recall", data=data)


def roc_curve(fpr: np.ndarray, tpr: np.ndarray) -> None:
    """Takes false-positive and true-positives rates, and
    plots a ROC curve with them.

    Args:
        fpr (np.ndarray): false positive rate
        tpr (np.ndarray): true positive rate
    """
    data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    plot = sns.lineplot(x="fpr", y="tpr", data=data)
    plot.set(xlabel="FPR", ylabel="TPR")
    plt.plot([0, 1], [0, 1], "k--")



def gridsearch_heatmap(
    gridresults: GridSearchCV,
    param_grid: Dict[str, List[float]],
    vmin: float = None,
    vmax: float = None,
    figsize: Tuple = (12, 12),
) -> None:
    """
    plots results from a gridsearch.
    Parameters:
      gridresults: A GridSearchCV object
      param_grid: a grid with two parameters
      vmin, vmax: used to cut of colors
      figsize: tuple, specifies plotsize
    """
    idx, col = ["param_" + [*param_grid.keys()][i] for i in range(2)]
    pivoted = pd.pivot_table(
        pd.DataFrame(gridresults.cv_results_),
        values="mean_test_score",
        index=idx,
        columns=col,
    )
    pivoted.index = ["{:.2f}".format(x) for x in pivoted.index]
    pivoted.columns = ["{:.2f}".format(x) for x in pivoted.columns]
    plt.figure(figsize=figsize)
    sns.heatmap(pivoted, vmin=vmin, vmax=vmax, annot=True)



