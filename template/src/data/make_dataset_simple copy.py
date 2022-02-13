
import os
import sys
from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, "..")

def scale(X: np.ndarray) -> np.ndarray:  # noqa: N803
    logger.info("Run standardscaler on data.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def set_class(
    y_train: np.ndarray, y_test: np.ndarray, cl: str
) -> Tuple[np.ndarray, np.ndarray]:
    """creates a train-test set, where every right class is True,
    and every other class is false

    Args:
        y_train (np.ndarray): numpy trainset
        y_test (np.ndarray): numpy testset
        cl (str): class to set. 

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    return (y_train == cl, y_test == cl)










