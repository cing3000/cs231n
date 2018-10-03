import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
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

    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
        
        z = np.dot(X[i, :], W)    # (1,D).(D,C) = (1,C)
        z -= np.max(z)
        z_exp = np.exp(z)
        norm = np.sum(z_exp)

        loss -= np.log(z_exp[y[i]]/norm)
        
        a = (z_exp/norm).reshape(C,1)
        a[y[i]] -= 1
        dW += (X[i] * a).T
    
    loss /= N
    loss += reg * np.sum(W**2)
    dW /= N
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
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
    N = X.shape[0]
    
    Z = np.dot(X, W)           # (N, D).(D, C) = (N,C)
    Z -= np.max(Z, axis=1, keepdims=True)
    Z_exp = np.exp(Z)
    A = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)

    loss = -np.sum(np.log(A[np.arange(N), y]))
    loss /= N
    loss += reg * np.sum(W**2)
    
    A[range(N), y] -= 1       # (N, C)
    dW = np.dot(X.T, A)
    dW /= N
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

