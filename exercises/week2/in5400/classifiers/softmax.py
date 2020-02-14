import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    print("X", X.shape)
    print("W", W.shape)
    C = W.shape[1]  # Number of classes
    D = W.shape[0]  # Number of dimensions
    N = X.shape[0]  # Number of examples in minibatch

    print(C, D, N)

    z = X @ W
    z -= np.max(z)
    print(z.shape)
    nx, ny = z.shape
    s = 0

    for k in range(N):

        for i in range(C):

            s += np.exp(z[k]) / np.exp(z[i])

            loss += -np.log(s)
    dW = loss * (1 - loss)

    # loss=[]
    #dw = []
    print(loss)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #loss = []
    #dW = []
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
