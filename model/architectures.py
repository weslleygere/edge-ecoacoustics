"""Compact multi-output regression networks for 299-frame acoustic features.

TensorFlow is imported only when a neural model is requested. Models have
three independent linear outputs; clipping to [0,1] is done only for reporting.
"""


def build_model(name: str, input_shape: tuple[int, int]):
    import tensorflow as tf

    if name not in ('mlp', 'cnn', 'dscnn'):
        raise ValueError(f'Unknown neural architecture: {name}')
    inputs = tf.keras.Input(shape=(*input_shape, 1), name='acoustic_features')
    if name == 'mlp':
        # Shared temporal summary: one scalar mean, standard deviation and
        # maximum per frequency bin, computed inside the exported network.
        x = tf.keras.layers.Reshape(input_shape)(inputs)
        avg = tf.keras.layers.GlobalAveragePooling1D()(x)
        maximum = tf.keras.layers.GlobalMaxPooling1D()(x)
        # A plain MLP gets an intentionally cheap, fixed statistical summary.
        x = tf.keras.layers.Concatenate()([avg, maximum])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
    else:
        x = inputs
        filters = (8, 16, 24)
        for n in filters:
            if name == 'cnn':
                x = tf.keras.layers.Conv2D(n, 3, strides=2, padding='same',
                                           activation='relu')(x)
            else:
                # First convolution also keeps the channel mixer simple.
                x = tf.keras.layers.SeparableConv2D(n, 3, strides=2,
                                                    padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='sigmoid', name='indices')(x)
    return tf.keras.Model(inputs, outputs, name=f'{name}_student')
