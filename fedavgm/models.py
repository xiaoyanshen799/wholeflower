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


def resnet18_keras(input_shape, num_classes, learning_rate):
    """Keras ResNet-18 adapted for CIFAR-style inputs.

    Uses 4 stages with block configuration [2, 2, 2, 2] and filter sizes
    [64, 128, 256, 512]. Downsampling is applied at the start of stages 2-4.
    """

    inputs = keras.Input(shape=input_shape)

    # Initial conv
    x = _resnet_layer(inputs=inputs, num_filters=64)

    # Define stages: (num_blocks, num_filters)
    stages = [
        (2, 64),
        (2, 128),
        (2, 256),
        (2, 512),
    ]

    for si, (num_blocks, num_filters) in enumerate(stages):
        for bi in range(num_blocks):
            strides = 1
            if si > 0 and bi == 0:
                strides = 2  # Downsample at stage start (except stage 1)

            y = _resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = _resnet_layer(inputs=y, num_filters=num_filters, activation=None)

            if si > 0 and bi == 0:
                # Projection to match dims when downsampling
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

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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


# -----------------------------------------------------------------------------
# NLP / Speech additions
# -----------------------------------------------------------------------------

class _TiedOutput(keras.layers.Layer):
    """Output layer tying weights to an Embedding layer (weight tying)."""

    def __init__(self, embedding_layer: keras.layers.Embedding, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.vocab_size = int(vocab_size)

    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=(self.vocab_size,), initializer="zeros")

    def call(self, inputs):
        logits = tf.linalg.matmul(inputs, tf.transpose(self.embedding_layer.embeddings))
        return logits + self.bias


def char_lstm2(input_shape, num_classes, learning_rate):
    """2-layer LSTM for character-level Shakespeare."""
    try:
        seq_len = int(input_shape[0])
    except Exception:
        seq_len = int(input_shape)

    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    emb = keras.layers.Embedding(input_dim=int(num_classes), output_dim=16, name="emb")
    x = emb(inputs)
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.LSTM(256, return_sequences=False)(x)
    outputs = _TiedOutput(embedding_layer=emb, vocab_size=num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="char_lstm2")
    optimizer = SGD(learning_rate=learning_rate, momentum=0.0)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def speech_cnn_small(input_shape, num_classes, learning_rate):
    """Small CNN for log-mel Speech Commands."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(int(num_classes), activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="speech_cnn_small")
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def embed_avg_mlp(
    input_shape,
    num_classes,
    learning_rate,
    *,
    vocab_size: int,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
):
    """Embedding -> masked average -> small MLP (StackOverflow tag prediction)."""
    try:
        seq_len = int(input_shape[0])
    except Exception:
        seq_len = int(input_shape)

    vocab_size = int(vocab_size)
    num_classes = int(num_classes)
    embedding_dim = int(embedding_dim)
    hidden_dim = int(hidden_dim)

    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="emb")(inputs)

    # Masked mean over non-PAD tokens (PAD_ID=0)
    mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)  # [B, L]
    mask = tf.expand_dims(mask, axis=-1)  # [B, L, 1]
    summed = tf.reduce_sum(emb * mask, axis=1)  # [B, D]
    denom = tf.reduce_sum(mask, axis=1)  # [B, 1]
    avg = summed / tf.maximum(denom, 1.0)

    x = keras.layers.Dense(hidden_dim, activation="relu")(avg)
    x = keras.layers.Dropout(float(dropout))(x)
    logits = keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="embed_avg_mlp")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def embed_avg_mlp_bce(
    input_shape,
    num_classes,
    learning_rate,
    *,
    vocab_size: int,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
):
    """Embedding -> masked average -> small MLP (multi-label BCE)."""
    try:
        seq_len = int(input_shape[0])
    except Exception:
        seq_len = int(input_shape)

    vocab_size = int(vocab_size)
    num_classes = int(num_classes)
    embedding_dim = int(embedding_dim)
    hidden_dim = int(hidden_dim)

    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="emb")(inputs)

    mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)  # [B, L]
    mask = tf.expand_dims(mask, axis=-1)  # [B, L, 1]
    summed = tf.reduce_sum(emb * mask, axis=1)  # [B, D]
    denom = tf.reduce_sum(mask, axis=1)  # [B, 1]
    avg = summed / tf.maximum(denom, 1.0)

    x = keras.layers.Dense(hidden_dim, activation="relu")(avg)
    x = keras.layers.Dropout(float(dropout))(x)
    logits = keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="embed_avg_mlp_bce")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
    )
    return model


def embed_bilstm_mlp(
    input_shape,
    num_classes,
    learning_rate,
    *,
    vocab_size: int,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.2,
):
    """Embedding -> BiLSTM -> small MLP (heavier than embed_avg_mlp)."""
    try:
        seq_len = int(input_shape[0])
    except Exception:
        seq_len = int(input_shape)

    vocab_size = int(vocab_size)
    num_classes = int(num_classes)
    embedding_dim = int(embedding_dim)
    lstm_units = int(lstm_units)
    hidden_dim = int(hidden_dim)

    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,  # PAD_ID=0
        name="emb",
    )(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)
    x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = keras.layers.Dropout(float(dropout))(x)
    logits = keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="embed_bilstm_mlp")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def embed_bilstm_mlp_bce(
    input_shape,
    num_classes,
    learning_rate,
    *,
    vocab_size: int,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.2,
):
    """Embedding -> BiLSTM -> small MLP (multi-label BCE)."""
    try:
        seq_len = int(input_shape[0])
    except Exception:
        seq_len = int(input_shape)

    vocab_size = int(vocab_size)
    num_classes = int(num_classes)
    embedding_dim = int(embedding_dim)
    lstm_units = int(lstm_units)
    hidden_dim = int(hidden_dim)

    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="emb",
    )(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)
    x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = keras.layers.Dropout(float(dropout))(x)
    logits = keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="embed_bilstm_mlp_bce")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
    )
    return model


def model_to_parameters(model):
    """Retrieve model weigths and convert to ndarrays."""
    return ndarrays_to_parameters(model.get_weights())
