from keras.layers import Dense, Input, Flatten
import tensorflow as tf
from ViT import create_vit_encoder

class Encoder:
    def __init__(self):
        pass

    def encode(self):
        pass

    def input_dims(self):
        pass


class FcEncoder(Encoder):
    def __init__(self):
        pass

    def encode(self, inputs_dims):
        inputs = Input(shape=inputs_dims, name='input')
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        return inputs, x

    def reshape_input(self, x):
        return x.reshape(x.shape[0], -1)

    def input_dims(self, x):
        return (x.shape[1],)


class ResnetEncoder(Encoder):

    def encode(self, inputs_dims):
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = Flatten()(base_model.output)
        return base_model.input, x

    # Pretrained resnet wants inputs to be of the shape (h, w, 3) with h, w >= 32
    # x.shape => (batch_size, 28, 28)
    def reshape_input(self, x):
        # x.shape => (batch_size, 32, 32)
        x = tf.pad(x, [[0, 0], [2, 2], [2, 2]])
        # x.shape => (batch_size, 32, 32, 1)
        x = tf.expand_dims(x, axis=3, name=None)
        # x.shape => (batch_size, 32, 32, 3)
        x = tf.repeat(x, 3, axis=3)
        return x

    def input_dims(self, x):
        return (x.shape[1:])

class ViTEncoder(Encoder):
    """ViTEncoder"""
    def __init__(self, arg=None):
        pass

    def encode(self, inputs_dims):
        model, inputs, representation = create_vit_encoder()
        return inputs, representation

    def reshape_input(self, x):
        x = tf.expand_dims(x, axis=3, name=None)
        return x

    def input_dims(self, x):
        return (x.shape[1:])