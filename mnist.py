from datetime import datetime
from pathlib import Path

import tcn
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_reshaped = 1/255 * x_train.reshape(-1, 28 * 28, 1)
    x_test_reshaped = 1/255 * x_test.reshape(-1, 28 * 28, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_reshaped, y_train)).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_reshaped, y_test)).shuffle(1000)
    return train_dataset, test_dataset


def train():
    levels = 6
    hidden_units = 25
    channel_sizes = [hidden_units] * levels

    model = tcn.build_model(sequence_lenght=28 * 28,
                            num_inputs=1,
                            num_classes=10,
                            num_channels=channel_sizes,
                            kernel_size=8)

    model.compile(optimizer="Adam",
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  loss=losses.SparseCategoricalCrossentropy())

    print(model.summary())

    train_dataset, test_dataset = load_dataset()

    model.fit(train_dataset.take(1000).batch(32),
              validation_data=test_dataset.take(1000).batch(32),
              callbacks=[TensorBoard(str(Path("logs") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")))],
              epochs=10)


if __name__ == '__main__':
    train()
