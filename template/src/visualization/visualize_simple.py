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


def predictions(
    X: np.ndarray, y: np.ndarray, model: protocol.GenericModel  # noqa: N803
) -> None:
    """Fit a model on X data and y target.
    generate an equally spaced x-axis between min and max values
    create a scatterplot of the data and a line of the model.

    Args:
        X (np.ndarray): data
        y (np.ndarray): target
        model ([type]): model with a .fit() function
    """
    model.fit(X, y)
    # make evenly spaced values over the input range
    newx = np.linspace(min(X), max(X), 100)
    yhat = model.predict(newx)
    plt.scatter(X, y, label="data")
    plt.plot(newx, yhat, label="model")
    plt.legend()


def impact_of_alpha(
    alphalist: List[float],
    pipe: pipeline.Pipeline,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,  # noqa: N803
) -> None:
    steps = [*pipe.named_steps]  # this gets the name of the steps
    model = steps[-1]  # this takes the last step, which is the model
    for alpha in alphalist:
        fit_params = {model + "__alpha": alpha}
        pipe.set_params(**fit_params)  # we set one of the alpha's from the alphaList
        pipe.fit(X, y)  # and fit the model
        coef = pipe.named_steps[model].coef_.T  # get the coeficients from the model
        x = range(len(coef))
        plt.scatter(x, coef, label=alpha)  # and plot them

    # some esthetics
    plt.legend(title="alpha")
    plt.title(model)
    plt.xticks(x, X.columns, rotation=90)
    plt.show()


def plot_contour(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    pipe: protocol.GenericModel,
    granularity: float = 0.1,
    grid_side: float = 0.5,
) -> None:
    X_train = np.array(X_train)  # noqa: N803, N806
    y_train = np.array(y_train)
    cm = plt.cm.PiYG  # let's change the colormap.

    # first, we get the min-max range over which we want to plot
    # this is the area for which we want to know the behavior of the model
    # we add some extra space with grid_side to the feature space.
    x0_min, x0_max = X_train[:, 0].min() - grid_side, X_train[:, 0].max() + grid_side
    x1_min, x1_max = X_train[:, 1].min() - grid_side, X_train[:, 1].max() + grid_side
    # we make a grid of coordinates
    xx, yy = np.meshgrid(
        np.arange(x0_min, x0_max, granularity), np.arange(x1_min, x1_max, granularity)
    )
    # and combine the grid into a new dataset.
    # this new dataset covers (with some granularity)
    # every point of the original dataset
    # this newx is equal to the featurespace we want to examine.
    newx = np.c_[xx.ravel(), yy.ravel()]

    # we make a prediction with the new dataset.
    # This will show us predictions over the complete featurespace.
    yhat = pipe.predict(newx)

    # and reshape the prediction, so that it will match our gridsize
    z = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=cm, alpha=0.5)
    data = pd.DataFrame({"x1": X_train[:, 0], "x2": X_train[:, 1]})
    sns.scatterplot(data=data, x="x1", y="x2", c=y_train, cmap=cm)


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


def compare_results(results: Dict[str, Dict[str, float]], ylim: float = None) -> None:
    """The result is a dictionary with results for training, testing and
    possibly validation. E.g.:

    >>> results['svm'] = {"train" : pipe.score(X_train, y_train),
                      "test" : pipe.score(X_test, y_test)}

    The dict will be melted into a format for comparing the results of different models.
        Args:
        results (Dict[str, Dict[str, float]]): Dict with

            {"model" :
                {"datasetname" : 0.8}
            }

            structure

        ylim ([type], optional): limits y-axis for better plotting. Defaults to None.
    """
    logger.info(f"Found results for {len(results.keys())} models")
    data = pd.DataFrame(results).reset_index()
    data = data.melt(id_vars="index")
    sns.barplot(x="index", y="value", hue="variable", data=data)
    plt.ylim(ylim, 1)
