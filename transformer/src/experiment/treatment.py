from noisenet import NoiseNet
import torch
import tensorflow as tf
import numpy as np


def odd_even_treatment(y):
    evens = tf.compat.v1.distributions.Bernoulli(probs=np.zeros_like(y) + 0.2).sample(sample_shape=1)
    odds = value = tf.compat.v1.distributions.Bernoulli(probs=np.zeros_like(y) + 0.8).sample(sample_shape=1)
    # either pick from evens or odds depending on y
    value = (((y % 2) + 1) % 2) * evens + (y % 2) * odds
    value = tf.squeeze(value)
    return value.numpy()


def pixel_count_treatment(x):
    x = x.reshape(x.shape[0], -1)
    black_pixels = np.equal(x, np.zeros_like(x)).sum(axis=1)
    black_pixels = (black_pixels / x.shape[1]) * 0.7
    value = tf.compat.v1.distributions.Bernoulli(probs=black_pixels).sample(sample_shape=1)
    value = tf.squeeze(value)
    return value.numpy()


def noisenet_treatment(x):
    network = NoiseNet()
    network.load_state_dict(torch.load('model.pth'))
    network.eval()

    # x should be of shape torch.Size([_, 1, 28, 28])
    input = torch.from_numpy(x).float()
    input = input.unsqueeze(1)
    input = input / 255
    output = network(input)
    log_outs, _ = torch.max(output, dim=1)
    # bound the values of log_outs between [0.1, 0.9] so that in Bernoulli all points get chance of treatment and not
    # treatment
    log_outs = ((log_outs - torch.min(log_outs)) / (torch.max(log_outs) - torch.min(log_outs))) * 0.8 + 0.1
    log_outs = log_outs.detach().numpy()
    value = tf.compat.v1.distributions.Bernoulli(probs=log_outs).sample(sample_shape=1)
    value = tf.squeeze(value)
    return value.numpy()
