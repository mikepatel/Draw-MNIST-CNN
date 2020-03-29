"""
Michael Patel
March 2020

Project description:
    Build an interactive MNIST classifier using pygame

File description:
    For model definitions
"""
################################################################################
# Imports
import tensorflow as tf

from parameters import *


################################################################################
# CNN
def build_cnn(input_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[4, 4],
        input_shape=input_shape,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=512,
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Dropout(
        rate=DROP_RATE
    ))

    model.add(tf.keras.layers.Dense(
        units=NUM_CLASSES,
        activation=tf.keras.activations.softmax
    ))

    return model
