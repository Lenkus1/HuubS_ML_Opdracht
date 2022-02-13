from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV

from src import protocol


# for 1_First_simple_model

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





