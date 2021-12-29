# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Tuple
from filelock import FileLock

import tensorflow as tf
from loguru import logger

logger.add("../reports/debug.log")



def create_generators(
    datagen_kwargs: Dict, dataflow_kwargs: Dict, datadir: Path, augment: bool = False
) -> Tuple[
    tf.keras.preprocessing.image.ImageDataGenerator,
    tf.keras.preprocessing.image.ImageDataGenerator,
]:
    """Makes datagenerators.
    More info at:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator


    Args:
        datagen_kwargs (Dict): Dictionary with arguments passed to
        ImageDataGenerator. E.g. rescale, validation_split

        dataflow_kwargs (Dict): Arguments for dataflow. Eg target_size, batch_size,
        interpolation.

        datadir (Path): location of the data

        augment (bool, optional): [description]. Defaults to False.

    Returns:
        Tuple[ tf.keras.preprocessing.image.ImageDataGenerator, tf.keras.preprocessing.image.ImageDataGenerator, ]: [description]
    """
    logger.info("Creating validation set data generator")
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    valid = valid_datagen.flow_from_directory(
        directory=datadir,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs,
    )

    logger.info("Creating train set data generator")
    if augment:
        # to squeeze more out your data, modify images.
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,  # rotate some degrees
            horizontal_flip=True,  # flip
            width_shift_range=0.2,  # shift
            height_shift_range=0.2,
            zoom_range=0.2,  # zoom
            **datagen_kwargs,
        )
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs
        )

    train = train_datagen.flow_from_directory(
        datadir, subset="training", shuffle=True, **dataflow_kwargs
    )
    return train, valid


def dataset_from_dir(datadir: Path, targetsize: Tuple[int, int]) -> Tuple:
    lock = str(datadir) + ".lock"
    with FileLock(lock):
        train = tf.keras.utils.image_dataset_from_directory(
            datadir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=targetsize,
            batch_size=32,
        )
        valid = tf.keras.utils.image_dataset_from_directory(
            datadir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=targetsize,
            batch_size=32,
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train = train.cache().prefetch(buffer_size=AUTOTUNE)
        valid = valid.cache().prefetch(buffer_size=AUTOTUNE)
    return train, valid
