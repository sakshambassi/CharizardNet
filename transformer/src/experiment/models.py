import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
import numpy as np


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def y_binary_classification_loss(concat_true, concat_pred):
    """
    Classification loss on y - used for MNIST causal effect
    """

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    # y0_pred = tf.dtypes.cast(y0_pred, tf.int64)
    # y1_pred = tf.dtypes.cast(y1_pred, tf.int64)

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    # y_pred = tf.dtypes.cast(y_pred, tf.int64)
    try:
        y_pred = (y_pred.numpy() + 0.001) / 1.002
        y_true = y_true.numpy()
        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')
        y_loss = np.sum(K.binary_crossentropy(y_true, y_pred))
    except AttributeError:
        y_pred = (y_pred + 0.001) / 1.002
        # y_true = y_true.numpy()
        # y_true = y_true.astype('float32')
        # y_pred = y_pred.astype('float32')
        y_loss = tf.reduce_sum(K.binary_crossentropy(y_true, y_pred))
    return y_loss


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def mnist_dragonnet_loss_binarycross(concat_true, concat_pred):
    """
    Total Classification loss - used for MNIST causal effect
    """
    return y_binary_classification_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


def make_dragonnet(encoder, input_dims, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    t_l1 = 0.
    t_l2 = reg_l2

    inputs, x = encoder.encode(input_dims)

    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2),
                           name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2),
                           name='y1_predictions')(y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model
