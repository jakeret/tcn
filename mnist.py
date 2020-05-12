from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import losses, metrics
from tensorflow.keras.callbacks import TensorBoard

import tcn


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_reshaped = 1/255 * x_train.reshape(-1, 28 * 28, 1)
    x_test_reshaped = 1/255 * x_test.reshape(-1, 28 * 28, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_reshaped, y_train)).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_reshaped, y_test)).shuffle(1000)
    return train_dataset, test_dataset


def train():
    depth = 6
    filters = 25
    block_filters = [filters] * depth

    model = tcn.build_model(sequence_length=28 * 28,
                            channels=1,
                            num_classes=10,
                            filters=block_filters,
                            kernel_size=8)

    model.compile(optimizer="Adam",
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  loss=losses.SparseCategoricalCrossentropy())

    print(model.summary())

    train_dataset, test_dataset = load_dataset()

    model.fit(train_dataset.batch(32),
              validation_data=test_dataset.batch(32),
              callbacks=[TensorBoard(str(Path("logs") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")))],
              epochs=10)


if __name__ == '__main__':
    train()
