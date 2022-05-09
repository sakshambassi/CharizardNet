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
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from metrics import *
from treatment import *
import time
import torch

from encoder import *

print("TF version is: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


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
        np.take(g, np.where(t.reshape(-1, ) == 1)).mean(),
        np.take(g, np.where(t.reshape(-1, ) == 0)).mean())
    print(var)

    acc = outcome_accuracy(t, q_t0, q_t1, y)
    print(f"Accuracy: {acc}")
    ate = average_treatment_effect(q_t0, q_t1)
    print(f"Average treatment effect: {ate}")
    att = average_treatment_effect_on_treated(q_t0, q_t1, t)
    print(f"Average treatment effect on treated: {att}")
    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'eps': eps}


def train_and_predict_dragons(t_train, t_test, y_train, y_test, x_train, x_test, encoder, args: argparse,
                              targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, val_split=0.2):
    batch_size = args.batch_size
    ratio = args.ratio
    epochs_adam = args.epochs_adam
    epochs_sgd = args.epochs_sgd

    verbose = 0
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    x_train = encoder.reshape_input(x_train)
    x_test = encoder.reshape_input(x_test)
    t_train = t_train.reshape(t_train.shape[0], -1)
    t_test = t_test.reshape(t_test.shape[0], -1)

    # if args.greene == True:
    #     y_train = y_train.to_device(device)
    #     y_test = y_test.to_device(device)
    #     x_train = x_train.to_device(device)
    #     x_test = x_test.to_device(device)
    #     t_train = t_train.to_device(device)
    #     t_test = t_test.to_device(device)

    input_dims = encoder.input_dims(x_train)

    # y_scaler = StandardScaler().fit(y_train)
    # y_train = y_scaler.transform(y_train)
    # y_test = y_scaler.transform(y_test)

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # print("I am here making dragonnet")

    dragonnet = make_dragonnet(encoder, input_dims, 0.01)
    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    yt_train = np.concatenate([y_train, t_train], 1)

    start_time = time.time()

    dragonnet.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics, run_eagerly=True)

    # if args.greene == True:
    #     dragonnet = dragonnet.to_device(device)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0)
    ]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=epochs_adam,
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
                  epochs=epochs_sgd,
                  batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = dragonnet.predict(x_test)
    ## yt_hat_test consists of these values: y0_predictions, y1_predictions, t_predictions, epsilons
    yt_hat_train = dragonnet.predict(x_train)

    test_outputs = _split_output(yt_hat_test, t_test, y_test, x_test)
    train_outputs = _split_output(yt_hat_train, t_train, y_train, x_train)
    K.clear_session()

    # print(f"y_test : {y_test}")
    # print(f"t_test : {t_test}")
    # print(f"yt_hat_test : {yt_hat_test}")

    return test_outputs, train_outputs


def create_treatment_values(x, y, treatment="noisenet"):
    print(f"Treatment used: {treatment}")
    if treatment == "noisenet":
        return noisenet_treatment(x)
    elif treatment == "odd-even":
        return odd_even_treatment(y)
    elif treatment == "pixel-count":
        return pixel_count_treatment(x)

##  old code
# def create_targets(y):
#     is_prime = lambda value: value in {2, 3, 4, 7}
#     return np.array([int(is_prime(value)) for value in y])

def create_targets(x, t):
    y1_arr = noisenet(x, 'y1')
    y0_arr = noisenet(x, 'y0')
    y_true = t * y1_arr + (1-t)*y0_arr
    return y_true

def run_mnist(args: argparse, output_dir: str):
    mnist = tf.keras.datasets.mnist
    (x_train, train_labels), (x_test, test_labels) = mnist.load_data()

    ## old code
    # y_train = create_targets(train_labels)
    # y_test = create_targets(test_labels)

    t_train = create_treatment_values(x_train, train_labels, args.treatment)
    t_test = create_treatment_values(x_test, test_labels, args.treatment)

    ## new code
    y_train = create_targets(x_train, t_train)
    y_test = create_targets(x_test, t_test)

    if args.dry_run:
        train_labels = train_labels[:args.dry_run_val]
        test_labels = test_labels[:args.dry_run_val]
        x_train = x_train[:args.dry_run_val]
        y_train = y_train[:args.dry_run_val]
        x_test = x_test[:args.dry_run_val]
        y_test = y_test[:args.dry_run_val]
        t_train = t_train[:args.dry_run_val]
        t_test = t_test[:args.dry_run_val]

    

    encoder = get_encoder(args.encoder)

    for is_targeted_regularization in [True, False]:
        print("Is targeted regularization: {}".format(is_targeted_regularization))
        test_outputs, train_output = train_and_predict_dragons(t_train, t_test, y_train, y_test, x_train, x_test,
                                                               args=args,
                                                               encoder=encoder,
                                                               targeted_regularization=is_targeted_regularization,
                                                               output_dir=output_dir,
                                                               knob_loss=mnist_dragonnet_loss_binarycross,
                                                               val_split=0.2)


def get_encoder(encoder_type: str):
    if encoder_type == "resnet":
        return ResnetEncoder()
    elif encoder_type == "vit":
        return ViTEncoder()
    else:
        return FcEncoder()


def turn_knob(args: argparse):
    output_dir = os.path.join(args.output_base_dir, args.knob)
    run_mnist(args, output_dir)


def main():
    parser = argparse.ArgumentParser()
    # original code has {epochs_adam: 100, epochs_sgd: 300, batch_size: 64}
    parser.add_argument('--knob', type=str, default='dragonnet')
    parser.add_argument('--data-base-dir', type=str, default="../dat/ihdp/csv")
    parser.add_argument('--output-base-dir', type=str, default="../result/ihdp")
    parser.add_argument('--dry-run', type=bool, default=True)
    parser.add_argument('--dry-run-val', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=1.)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-adam', type=int, default=100)
    parser.add_argument('--epochs-sgd', type=int, default=300)
    # Possible values of encoder are 'fc', 'resnet', 'vit'
    parser.add_argument('--encoder', type=str, default='resnet')
    parser.add_argument('--greene', type=bool, default=True)
    # Possible values of treatment are 'odd-even', 'noisenet', 'pixel-count'
    parser.add_argument('--treatment', type=str, default="odd-even")

    args = parser.parse_args()

    print(f'Model used: {args.encoder}')
    print(f'Is only trained on small data? : {args.dry_run}')
    # if args.greene == True:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    turn_knob(args)


if __name__ == '__main__':
    main()
