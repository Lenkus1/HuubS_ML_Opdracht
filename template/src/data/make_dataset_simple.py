
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

def data_set_maken(datadir_processed
) -> None:
    # data en label definiëren
    data = [] 
    labels = []

    # met behulp van loop alle foto's in de 10 subdirectories doorlopen
    for f in sorted(os.listdir(datadir_processed)):
        folder = os.path.join(datadir_processed, f)
        if os.path.isdir(folder):
        
            for i in sorted(os.listdir(folder)):
                image=tf.keras.preprocessing.image.load_img(folder+'/'+i, color_mode='grayscale', # omzetten naar 1 channel voor lineair model
                target_size= (64,64))
                image=np.array(image)
                data.append(image)
                labels.append(f)    # foldernaam als label

    data = np.array(data)
    labels = np.array(labels) 

    return data, labels








