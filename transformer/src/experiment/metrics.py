import sklearn
import numpy as np
import tensorflow as tf


def get_accuracy(t, y_0, y_1, y_true):
    t = tf.squeeze(t)
    y_pred = round(t * y_1 + (1 - t) * y_0)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    return acc


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
    mean_y0_given_t1 = np.array(y0_given_t1).mean()
    mean_y0_given_t0 = np.array(y0_given_t0).mean()
    return (0 if np.isnan(mean_y0_given_t1) else mean_y0_given_t1) - \
           (0 if np.isnan(mean_y0_given_t0) else mean_y0_given_t0)


def average_treatment_effect(y0, y1):
    '''
    E[Y1 - Y0]
    '''
    ate = (y1 - y0).mean()
    return ate


def average_treatment_effect_on_treated(y0, y1, t):
    '''
    E[Y1 - Y0 | T = 1]
    '''
    t = tf.squeeze(t)
    treated_indices = np.where(t == 1)
    y0_when_treated = y0[treated_indices]
    y1_when_treated = y1[treated_indices]
    ate = (y1_when_treated - y0_when_treated).mean()
    return ate
