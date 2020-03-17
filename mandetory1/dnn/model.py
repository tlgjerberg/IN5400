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

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1

    params = {}

    L = len(conf['layer_dimensions'])

    for l in range(1, L):

        b = np.zeros((conf['layer_dimensions'][l], 1))
        W = np.random.normal(
            0.0,
            2 / conf['layer_dimensions'][l],
            size=[conf['layer_dimensions'][l - 1],
                  conf['layer_dimensions'][l]])

        params.update({f'W_{l}': W})
        params.update({f'b_{l}': b})

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        return np.where(Z >= 0, Z, 0)

    if activation_function == 'leaky relu':
        return np.where(Z >= 0, Z, Z * 0.01)

    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """

    Z_ = Z - np.max(Z)
    t = Z_ - np.log(np.sum(np.exp(Z_), axis=0))

    return np.exp(t)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)

    Y_proposed = np.zeros((conf['layer_dimensions'][-1], X_batch.shape[0]))

    L = len(conf['layer_dimensions']) - 1

    features = {}
    features.update({'A_0': X_batch})

    for l in range(1, L):

        Z = np.dot(params[f'W_{l}'].T, features[f'A_{l-1}']) + params[f'b_{l}']
        features.update({f'Z_{l}': Z})
        A = activation(Z, 'relu')
        features.update({f'A_{l}': A})

    Z = np.dot(params[f'W_{L}'].T, features[f'A_{L-1}']) + params[f'b_{L}']
    A = softmax(Z)
    features.update({f'Z_{L}': Z})
    features.update({f'A_{L}': A})
    Y_proposed = A

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3

    m = Y_proposed.shape[1]
    cost = -1. / m * np.sum(Y_reference * np.log(Y_proposed))

    num_correct = 0
    for i in range(m):

        if np.argmax(Y_proposed[:, i]) == np.argmax(Y_reference[:, i]):
            num_correct += 1

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        return np.where(Z >= 0, 1, 0)

    if activation_function == 'leaky relu':
        return np.where(Z >= 0, 1, 0.01)

    else:
        print("Error: Unimplemented derivative of activation function: {}",
              activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)

    grad_params = {}

    # Backpropagation of output layer
    L = len(conf['layer_dimensions']) - 1
    m = Y_proposed.shape[1]
    J_L = Y_proposed - Y_reference

    grad_W = (1 / m) * features[f'A_{L-1}'] @ J_L.T
    grad_b = (1 / m) * J_L @ np.ones(m)

    grad_params.update({f'grad_W_{L}': grad_W})
    grad_params.update({f'grad_b_{L}': grad_b.reshape(len(grad_b), 1)})

    # Backpropagation of hidden layers
    for l in range(L - 1, 0, -1):

        # Fetching layers and activations
        Z = features[f'Z_{l}']
        A = features[f'A_{l-1}']

        g_prime = activation_derivative(Z, 'relu')
        J = np.multiply(g_prime, params[f'W_{l+1}'] @ J_L)

        grad_W = 1 / m * A @ J.T
        grad_b = 1 / m * J @ np.ones(m)
        grad_params.update({f'grad_W_{l}': grad_W})
        grad_params.update({f'grad_b_{l}': grad_b.reshape(len(grad_b), 1)})

        J_L = J

    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    updated_params = {}

    lr = conf['learning_rate']

    for key in params:

        updated_params.update(
            {key: params[key] - lr * grad_params[f'grad_{key}']})

    return updated_params
