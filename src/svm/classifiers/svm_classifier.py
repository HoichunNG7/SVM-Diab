import numpy as np
from svm.classifiers.linear_svm import *


class SVMClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    loss_history = []
    N = np.size(y)
    D = X.shape[1]

    if self.W is None:
      self.W = np.random.randn(D + 1, ) * 0.0001

    for i in range(num_iters):
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch.                                                              #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
    
      mask = np.random.choice(N, batch_size, replace=True)
      X_batch = X[mask]
      y_batch = y[mask]
     
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################
    

      # evaluate loss and gradient

      loss, grad = linear_svm_loss_vectorized(self.W, X_batch, y_batch, reg)
      loss_history.append(loss)
        
      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################

      self.W = self.W - learning_rate * grad

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: N x D array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    N = X.shape[0]
    y_pred = np.zeros(N)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################

    X = np.c_[X, np.ones(N)] # new column corresponding to intercept

    for i in range(N):
      result = np.sum(np.dot(X[i], self.W))
      if result > 0:
        y_pred[i] = 1
      else:
        y_pred[i] = -1
    
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W;
    """
    pass


class LinearSVM(SVMClassifier):
  """ A subclass that uses the linear SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return linear_svm_loss_vectorized(self.W, X_batch, y_batch, reg)
