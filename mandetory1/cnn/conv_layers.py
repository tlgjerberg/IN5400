#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Implementation of convolution forward and backward pass"""

import numpy as np


def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_w, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1
    # Should have shape (batch_size, num_filters, height_y, width_y)
    output_layer = None

    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
    width_y = 1 + (width_x + 2 * pad_size - width_w) // stride

    npad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))

    input_padded = np.pad(input_layer, npad)

    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))

    K = pad_size

    for cx in range(channels_x):

        for cw in range(channels_w):

            for nf in range(num_filters):

                for p in range(0, height_x):

                    for q in range(0, width_x):

                        for r in range(-K, K):

                            for s in range(-K, K):

                                output_layer[:, nf, p, q] += \
                                    np.dot(input_padded[:, cx, p + r, q + s],
                                           weight[nf, cw, r, s])
                            # print(output_layer[:, nf, p, q])

        # print(input_padded[:, cx, p + r, q + s] * weight[nf, cw, r, s])

    print(output_layer[0, 0, 0, :])
    print(output_layer[0, 0, 3, :])

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient, weight_gradient, bias_gradient = None, None, None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
