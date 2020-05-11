from tensorflow.keras import Model, Input
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):

    def __init__(self,
                 out_channels,
                 kernel_size,
                 dilation_rate,
                 padding='causal',
                 dropout_rate=0.0,
                 activation="relu",
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.out_channels = out_channels
        filters = out_channels

        self.causal_conv_1 = layers.Conv1D(filters=filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           padding=padding)
        self.weight_norm_1 = layers.LayerNormalization()
        self.dropout_1 = layers.SpatialDropout1D(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.causal_conv_2 = layers.Conv1D(filters=filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           padding=padding)
        self.weight_norm_2 = layers.LayerNormalization()
        self.dropout_2 = layers.SpatialDropout1D(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)

        self.activation_3 = layers.Activation(activation)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels == self.out_channels:
            self.skip_conv = None
        else:
            self.skip_conv = layers.Conv1D(filters=self.out_channels,
                                           kernel_size=1)

        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if self.skip_conv is None:
            skip = inputs
        else:
            skip = self.skip_conv(inputs)

        x = self.causal_conv_1(inputs)
        x = self.weight_norm_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)

        x = self.causal_conv_2(x)
        x = self.weight_norm_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x, training=training)

        x = self.activation_3(x + skip)
        return x


def build_model(sequence_lenght, num_inputs, num_channels, num_classes, kernel_size):
    inputs = Input(shape=(sequence_lenght, num_inputs), name="inputs")

    x = inputs

    num_levels = len(num_channels)

    receptive_field_size = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
    print(f"Input sequence lenght: {sequence_lenght}, model receptive field: {receptive_field_size}")

    for i in range(num_levels):
        dilation_size = 2 ** i
        out_channels = num_channels[i]

        x = ResidualBlock(out_channels,
                          kernel_size=kernel_size,
                          dilation_rate=dilation_size,
                          name=f"residual_block_{i}")(x)

    x = layers.Lambda(lambda tt: tt[:, -1, :])(x)
    outputs = layers.Dense(num_classes,
                           activation="softmax",
                           name="output")(x)

    model = Model(inputs, outputs, name="tcn")

    return model
