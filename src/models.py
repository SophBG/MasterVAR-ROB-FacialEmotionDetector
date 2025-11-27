import tensorflow as tf
from tensorflow.keras import layers, models

def mini_xception(input_shape=(48,48,1), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(8, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    def residual_block(x, filters):
        res = x
        x = layers.SeparableConv2D(filters, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        # projection
        res = layers.Conv2D(filters, (1,1), strides=(2,2), padding='same')(res)
        x = layers.add([x, res])
        return x

    x = residual_block(x, 16)
    x = residual_block(x, 32)
    x = residual_block(x, 64)
    x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model
