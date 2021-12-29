import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger

# from linting import Any

logger.add("../reports/debug.log")

def plot_batch(
    generator: tf.keras.preprocessing.image.ImageDataGenerator, grid: int = 9
) -> None:
    inv_map = {v: k for k, v in generator.class_indices.items()}

    plt.figure(figsize=(30, 10))
    image, label = next(generator)
    logger.info(f"image shape: {image.shape}")
    logger.info(f"label shape: {label.shape}")
    gridn = int(np.ceil(np.sqrt(grid)))
    for i in range(grid):
        plt.subplot(gridn, gridn, i + 1)
        plt.imshow(image[i])
        plt.title(inv_map[np.argmax(label[i])])
        plt.axis("off")