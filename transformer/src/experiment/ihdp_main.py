# test
import os
import numpy as np
import math

from models import *
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from idhp_data import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from encoder import *


def pre_treatment_bias(q_t0, y, t):
    '''
    bias = E[Y0 | T = 1] - E[Y0 | T = 0]
         = E[counter factuals/predicted y/q_t0 when T = 1] - E[y when t = 0]
    '''
    y0_given_t1 = list()
    y0_given_t0 = list()

    for index, treatment in enumerate(t):
        if treatment == 1:
            y0_given_t1.append(q_t0[index])
        else:
            y0_given_t0.append(y[index])

    return np.array(y0_given_t1).mean() - np.array(y0_given_t0).mean()


def _split_output(yt_hat, t, y, x):
    # q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    # q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())

    q_t0 = yt_hat[:, 0]
    q_t1 = yt_hat[:, 1]
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    # y = y_scaler.inverse_transform(y.copy())
    bias = pre_treatment_bias(q_t0, y, t)
    print(f"pre treatment bias = {bias}")

    # had to change this since the tensors have now become eager tensors and previous methods dont work
    var = "average propensity for treated: {} and untreated: {}".format(
        tf.gather(g, tf.where(tf.squeeze(t) == 1.)).mean(),
        tf.gather(g, tf.where(tf.squeeze(t) == 0)).mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'eps': eps}


def train_and_predict_dragons(t_train, t_test, y_train, y_test, x_train, x_test, encoder, targeted_regularization=True,
                              output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    x_train = encoder.reshape_input(x_train)
    x_test = encoder.reshape_input(x_test)
    t_train = t_train.reshape(t_train.shape[0], -1)
    t_test = t_test.reshape(t_test.shape[0], -1)

    input_dims = encoder.input_dims(x_train)

    # y_scaler = StandardScaler().fit(y_train)
    # y_train = y_scaler.transform(y_train)
    # y_test = y_scaler.transform(y_test)

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    print("I am here making dragonnet")


    dragonnet = make_dragonnet(encoder, input_dims, 0.01)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    yt_train = np.concatenate([y_train, t_train], 1)

    import time
    start_time = time.time()

    dragonnet.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics, run_eagerly=True)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0)
    ]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-6
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(learning_rate=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                      metrics=metrics)

    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = dragonnet.predict(x_test)
    ## yt_hat_test consists of these values: y0_predictions, y1_predictions, t_predictions, epsilons
    yt_hat_train = dragonnet.predict(x_train)

    test_outputs = _split_output(yt_hat_test, t_test, y_test, x_test)
    train_outputs = _split_output(yt_hat_train, t_train, y_train, x_train)
    K.clear_session()

    print(f"y_test : {y_test}")
    print(f"t_test : {t_test}")
    print(f"yt_hat_test : {yt_hat_test}")

    return test_outputs, train_outputs


def create_treatment_values(x):
    value = tf.compat.v1.distributions.Bernoulli(probs=0.5).sample(sample_shape=x.shape)
    return value


def create_targets(y):
    is_prime = lambda value: value in {2, 3, 4, 7}
    return np.array([int(is_prime(value)) for value in y])


def run_mnist(args: argparse, output_dir: str):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = create_targets(y_train)
    y_test = create_targets(y_test)

    if args.dry_run:
        x_train = x_train[:10]
        y_train = y_train[:10]
        x_test = x_test[:10]
        y_test = y_test[:10]

    t_train = create_treatment_values(y_train)
    t_test = create_treatment_values(y_test)

    encoder = get_encoder(args.encoder)

    for is_targeted_regularization in [True, False]:
        print("Is targeted regularization: {}".format(is_targeted_regularization))
        test_outputs, train_output = train_and_predict_dragons(t_train, t_test, y_train, y_test, x_train, x_test,
                                                               encoder = encoder,
                                                               targeted_regularization=is_targeted_regularization,
                                                               output_dir=output_dir,
                                                               knob_loss=mnist_dragonnet_loss_binarycross,
                                                               ratio=args.ratio,
                                                               dragon=args.knob,
                                                               val_split=0.2,
                                                               batch_size=args.batch_size)

# TODO: Add plug for ResNet and Transformer
def get_encoder(encoder_type: str):
    if encoder_type == "fc":
        return FcEncoder()
    return FcEncoder()

def turn_knob(args: argparse):
    output_dir = os.path.join(args.output_base_dir, args.knob)
    run_mnist(args, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knob', type=str, default='dragonnet')
    parser.add_argument('--data-base-dir', type=str, default="../dat/ihdp/csv")
    parser.add_argument('--output-base-dir', type=str, default="../result/ihdp")
    parser.add_argument('--dry-run', type=bool, default=True)
    parser.add_argument('--ratio', type=float, default=1.)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--encoder', type=str, default='fc')

    args = parser.parse_args()
    turn_knob(args)


if __name__ == '__main__':
    main()
