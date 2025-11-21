"""CNN model architecture."""

from flwr.common import ndarrays_to_parameters
from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import keras
from tensorflow.nn import local_response_normalization  
import tensorflow as tf


# -----------------------------------------------------------------------------
# Keras implementation for CIFAR-10 ResNet-20 (used by FlowerClient)
# -----------------------------------------------------------------------------


def _resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
):
    """2-D Convolution-BN(optional)-Activation layer builder"""
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(inputs)
    if batch_normalization:
        x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def resnet20_keras(input_shape, num_classes, learning_rate):
    """Keras ResNet-20 v1 for CIFAR-10 (3×{3,3,3})."""

    inputs = keras.Input(shape=input_shape)
    num_filters = 16

    # First conv
    x = _resnet_layer(inputs=inputs, num_filters=num_filters)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(3):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = _resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = _resnet_layer(inputs=y, num_filters=num_filters, activation=None)

            if stack > 0 and res_block == 0:
                # Linear projection to match dims
                x = _resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )

            x = keras.layers.add([x, y])
            x = keras.layers.Activation("relu")(x)

        num_filters *= 2  # Double filters each stack

    # Final classification layer
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def cnn(input_shape, num_classes, learning_rate):
    """CNN Model from (McMahan et. al., 2017).

    Communication-efficient learning of deep networks from decentralized data
    """
    input_shape = tuple(input_shape)

    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(
                384, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(
                192, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def tf_example(input_shape, num_classes, learning_rate):
    """CNN Model from TensorFlow v1.x example.

    This is the model referenced on the FedAvg paper.

    Reference:
    https://web.archive.org/web/20170807002954/https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    """
    input_shape = tuple(input_shape)

    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.Lambda(
                local_response_normalization,
                arguments={
                    "depth_radius": 4,
                    "bias": 1.0,
                    "alpha": 0.001 / 9.0,
                    "beta": 0.75,
                },
            ),
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
            ),
            keras.layers.Lambda(
                local_response_normalization,
                arguments={
                    "depth_radius": 4,
                    "bias": 1.0,
                    "alpha": 0.001 / 9.0,
                    "beta": 0.75,
                },
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(
                384, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(
                192, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def _mobilenet_v2(
    input_shape,
    num_classes,
    learning_rate,
    alpha: float,
) -> keras.Model:
    """Build a MobileNetV2 backbone adapted to small grayscale inputs."""
    input_shape = tuple(input_shape)
    if len(input_shape) == 2:
        input_shape = (*input_shape, 1)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    height, width = input_shape[0], input_shape[1]
    channels = input_shape[2]

    # Ensure minimum spatial size expected by MobileNetV2
    target_hw = 96
    if height < target_hw or width < target_hw:
        x = keras.layers.Resizing(target_hw, target_hw, interpolation="bilinear")(x)
        height, width = target_hw, target_hw

    # MobileNetV2 expects 3 channels
    if channels == 1:
        x = keras.layers.Concatenate(axis=-1)([x, x, x])
    elif channels != 3:
        # Project to 3 channels if original data has a different channel count
        x = keras.layers.Conv2D(3, kernel_size=1, padding="same", activation=None)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(height, width, 3),
        include_top=False,
        weights=None,
        alpha=alpha,
    )
    base.trainable = True

    features = base(x, training=True)
    features = keras.layers.GlobalAveragePooling2D()(features)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    weight_decay = 1e-4
    try:
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    except TypeError:
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
        )

        def l2_regularizer():
            return weight_decay * tf.add_n(
                [
                    tf.nn.l2_loss(v)
                    for v in model.trainable_variables
                    if "bias" not in v.name and "beta" not in v.name
                ]
            )

        model.add_loss(l2_regularizer)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def mobilenet_v2_075(input_shape, num_classes, learning_rate):
    """MobileNetV2 with width multiplier 0.75."""
    return _mobilenet_v2(input_shape, num_classes, learning_rate, alpha=0.75)


def mobilenet_v2_100(input_shape, num_classes, learning_rate):
    """MobileNetV2 with width multiplier 1.0."""
    return _mobilenet_v2(input_shape, num_classes, learning_rate, alpha=1.0)


def model_to_parameters(model):
    """Retrieve model weigths and convert to ndarrays."""
    return ndarrays_to_parameters(model.get_weights())
