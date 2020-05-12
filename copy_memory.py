from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.callbacks import TensorBoard

import tcn


def load_dataset(batch_size, T):
    x_train, y_train = generate_copy_sequence(batch_size, T)
    x_test, y_test = generate_copy_sequence(batch_size, T)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000)
    return train_dataset, test_dataset


def generate_copy_sequence(batch_size, sequence_length):
    x = np.zeros((batch_size, sequence_length))
    copy_sequence = np.random.randint(0, 8, (batch_size, 10))
    x[:, :10] = copy_sequence
    x[:, -11:] = 9

    y = np.zeros_like(x)
    y[:, -10:] = copy_sequence
    return x, y


def train():
    depth = 6
    filters = 25
    block_filters = [filters] * depth
    sequence_length = 601

    train_dataset, test_dataset = load_dataset(30000, sequence_length)

    model = tcn.build_model(sequence_length=sequence_length,
                            channels=1,
                            num_classes=10,
                            filters=block_filters,
                            kernel_size=8,
                            return_sequence=True)

    model.compile(optimizer=optimizers.RMSprop(lr=5e-4, clipnorm=1.),
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  loss=losses.SparseCategoricalCrossentropy())

    print(model.summary())

    model.fit(train_dataset.batch(32),
              validation_data=test_dataset.batch(32),
              callbacks=[TensorBoard(str(Path("logs") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")))],
              epochs=10)


if __name__ == '__main__':
    train()
