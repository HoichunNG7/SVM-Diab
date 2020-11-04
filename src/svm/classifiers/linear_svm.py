import numpy as np
from random import shuffle


def linear_svm_loss_vectorized(W, X, y, reg):
  """
  Linear SVM loss function, vectorized implementation.
  Inputs have dimension D and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D+1, ) containing weights and the intercept('b').
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the linear SVM         #
  # loss, storing the result in dW.                                            #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  N = np.size(y)
  D_plus_one = np.size(W)
  temp_loss_sum = 0.0

  X = np.c_[X, np.ones(N)] # new column corresponding to intercept
 
  for i in range(N):
    cand = y[i] * np.sum(np.dot(X[i], W))
    if cand < 1: # support vectors
      temp_loss_sum += 1 - cand
      dW = dW - y[i] * X[i] / N # update gradient
    else:
      pass
  
  loss = temp_loss_sum / N

  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
