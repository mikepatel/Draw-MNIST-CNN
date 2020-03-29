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
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import *


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

    # reshape images
    train_images = train_images.reshape(-1, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)
    validation_images = validation_images.reshape(-1, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)
    test_images = test_images.reshape(-1, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)

    # shapes of data sets
    print(f'Shape of training images: {train_images.shape}')
    print(f'Shape of training labels: {train_labels.shape}')
    print(f'Shape of validation images: {validation_images.shape}')
    print(f'Shape of validation labels: {validation_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #
    input_shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)

    model = build_cnn(input_shape)
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=train_images.shape[0] // BATCH_SIZE,
        validation_data=(validation_images, validation_labels)
    )

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1.1])
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(output_dir + "\\Training Accuracy")
