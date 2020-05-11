from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import losses, metrics
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

import tcn


def load_dataset(num_words, sequence_length):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    x_train = sequence.pad_sequences(x_train, maxlen=sequence_length)
    x_test = sequence.pad_sequences(x_test, maxlen=sequence_length)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000)
    return train_dataset, test_dataset


def train():
    num_words = 20000
    sequence_length = 100
    depth = 6
    filters = 64
    channels = 128
    block_filters = [filters] * depth
    num_classes = 2

    inputs = layers.Input(shape=(sequence_length, ), name="inputs")
    x = layers.Embedding(num_words, channels)(inputs)
    x = tcn.TCN(block_filters, kernel_size=8)(x)
    outputs = layers.Dense(num_classes,
                           activation="softmax",
                           name="output")(x)

    model = Model(inputs, outputs)

    model.compile(optimizer="Adam",
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  loss=losses.SparseCategoricalCrossentropy())

    print(model.summary())

    train_dataset, test_dataset = load_dataset(num_words, sequence_length)

    model.fit(train_dataset.batch(32),
              validation_data=test_dataset.batch(32),
              callbacks=[TensorBoard(str(Path("logs") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")))],
              epochs=5)


if __name__ == '__main__':
    train()
