"""
Michael Patel
March 2020

Project description:
    Build an interactive MNIST classifier using pygame

File description:
    For model training and preprocessing
"""
################################################################################
# Imports
import os
from datetime import datetime

import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    # TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # create validation set
    half_point = int(len(test_images) / 2)
    validation_images = test_images[:half_point]
    validation_labels = test_labels[:half_point]

    test_images = test_images[half_point:]
    test_labels = test_labels[half_point:]

    # shapes of data sets
    print(f'Shape of training images: {train_images.shape}')
    print(f'Shape of training labels: {train_labels.shape}')
    print(f'Shape of validation images: {validation_images.shape}')
    print(f'Shape of validation labels: {validation_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #
