import tensorflow as tf
from tensorflow.keras import layers


def resnet50v2_classification_model(input_shape=(228, 228, 3),
                                    filters=64,
                                    num_classes=3,
                                    ):
    version = 2
    new_in = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=input_shape,
                                                  pooling=None,
                                                  classes=num_classes,
                                                  # **kwargs,
                                                  )
    base_model.trainable = True
    base_model.layers.pop(0)
    base_output = base_model(new_in)
    x = layers.GlobalAveragePooling2D()(base_output)
    x = layers.Dense(filters * 8, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(filters * 4, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(int(filters / 2), activation='relu')(x)
    x_out = layers.Dense(num_classes, activation='softmax')(x)
    keras_model = tf.keras.models.Model(inputs=new_in, outputs=x_out)
    name_of_model = f"ResNet50V2_{version}_{input_shape[0]}x{input_shape[1]}"
    return keras_model, name_of_model


def resnet50v2_original_model(input_shape=(228, 228, 3),
                              filters=64,
                              num_classes=3,
                              ):
    version = 2
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=input_shape,
                                                  pooling="avg",
                                                  classes=num_classes,
                                                  classifier_activation="softmax",
                                                  )
    base_model.trainable = True
    x = layers.Dense(filters * 8, activation='relu')(base_model.output)
    x = layers.Dropout(0.40)(x)
    # x = layers.Dense(filters * 4, activation='relu')(x)
    # x = layers.Dropout(0.35)(x)
    x = layers.Dense(int(filters / 2), activation='relu')(x)
    x_out = layers.Dense(num_classes, activation='softmax')(x)
    keras_model = tf.keras.models.Model(inputs=base_model.input, outputs=x_out)
    name_of_model = f"ResNet50V2_orig_{version}_{input_shape[0]}x{input_shape[1]}"
    return keras_model, name_of_model


def sepconv2d(input_shape, num_classes):
    """
    Args:
        input_shape (tuple):    input shape
        num_classes (int):      qty of classes

    Returns:
        keras_model (object):   keras model object
        name_of_model (str):    name of the model
    """
    inputs = layers.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)
    x = inputs
    # Entry block
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)  # added momentum [ 0.9 ]
    x = layers.Activation("relu")(x)

    x = layers.Dropout(0.5)(x)  # <= added dropout layer [ ]

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)  # added momentum [ 0.9 ]
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # added momentum [ 0.9 ]
        x = layers.Activation("relu")(x)

        # x = layers.Dropout(0.5)(x) #<= added dropout layer [ ]

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # added momentum [ 0.9 ]
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)  # added momentum [ 0.9 ]
    x = layers.Activation("relu")(x)

    x = layers.Dropout(0.5)(x)  # <= added dropout layer [ 0.2 0.5 ]

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.6)(x)  # <= dropout [ 0.5, 0.52, 0.5, 0.6 ,0.6 ]
    outputs = layers.Dense(units, activation=activation)(x)
    keras_model = tf.keras.Model(inputs, outputs)
    name_of_model = f"sepconv2d_{input_shape[0]}x{input_shape[0]}"
    return keras_model, name_of_model
