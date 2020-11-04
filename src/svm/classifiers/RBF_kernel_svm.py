import numpy as np
from random import shuffle

class KernelSVMClassifier(object):

  def __init__(self, X=None, y=None):
    self.b = 0.0 # intercept
    self.kernel_matrix = None
    self.X = X
    self.y = y
    self.sigma_square = 0.0 # bandwidth

    if self.X is not None:
      N = X.shape[0]
      self.Alpha = np.random.randn(N, ) * 0.001
    else:
      self.Alpha = None

  def train(self, X=None, y=None, learning_rate=1e-3, C=1, num_iters=2000, verbose=False):
    if X is not None:
      self.X = X
    if y is not None:
      self.y = y

    loss_history = []
    N = self.X.shape[0]
    D = self.X.shape[1]

    if self.Alpha is None:
      self.Alpha = np.random.randn(N, ) * 0.001
    if self.kernel_matrix is None:
      self.update_kernel_matrix()
    
    for i in range(num_iters):
      sample = np.random.randint(0, N, 1)
      index = sample[0]

      loss, grad = self.RBF_kernel_svm_loss(index, C)
      loss_history.append(loss)

      self.Alpha = self.Alpha - learning_rate * grad

    return loss_history


  def predict(self, X):
    N = X.shape[0]
    y_pred = np.zeros(N)

    for i in range(N):
      result = 0
      for j in range(self.X.shape[0]):
        result += self.kernel_function(X[i], self.X[j]) * self.Alpha[j]
      
      result += self.b # add intercept
      if result > 0:
        y_pred[i] = 1
      else:
        y_pred[i] = -1
    
    return y_pred

  def RBF_kernel_svm_loss(self, X_index, C):
    """
    RBF kernel Soft-SVM loss function.
    """
    loss = 0.0
    dAlpha = np.dot(self.kernel_matrix, self.Alpha) # initialize the gradient as zero

    temp_loss_sum = np.dot(self.kernel_matrix[X_index], self.Alpha) * self.y[X_index]
    if temp_loss_sum < 1: # support vectors
      loss = C * (1 - temp_loss_sum)
      dAlpha = dAlpha - C * self.y[X_index] * self.kernel_matrix[X_index]
    else:
      pass

    return loss, dAlpha

  def update_kernel_matrix(self):
    N = self.X.shape[0]
    dist_square_matrix = np.zeros((N, N))

    for i in range(N):
      for j in range(i+1):
        vec = self.X[i] - self.X[j]
        dist_square = np.dot(vec, vec)
        dist_square_matrix[i][j] = dist_square
        dist_square_matrix[j][i] = dist_square

    self.sigma_square = np.median(dist_square_matrix) # median trick

    self.kernel_matrix = np.exp((-1 / (2 * self.sigma_square)) * dist_square_matrix)
    return

  def kernel_function(self, X_i, X_j):
    vec = X_i - X_j

    dist_square = np.dot(vec, vec)
    kernel_value = np.exp((-1 / (2 * self.sigma_square)) * dist_square)

    return kernel_value

  def get_intercept(self): # calculate a valid intercept(self.b) for the model
    self.b = 0.0
    return