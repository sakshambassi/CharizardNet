from noisenet import NoiseNet
import torch
import tensorflow as tf


def odd_even_treatment(y):
    return y % 2


def noisenet_treatment(x):
    network = NoiseNet()
    network.load_state_dict(torch.load('model.pth'))
    network.eval()

    # x should be of shape torch.Size([_, 1, 28, 28])
    input = torch.from_numpy(x).float()
    input = input.unsqueeze(1)
    input = input/255
    output = network(input)
    log_outs, _ = torch.max(output, dim=1)
    log_outs = abs(log_outs)
    log_outs = log_outs / 2
    log_outs = log_outs.detach().numpy()
    value = tf.compat.v1.distributions.Bernoulli(probs=log_outs).sample(sample_shape=1)
    value = tf.squeeze(value)
    return value.numpy()
