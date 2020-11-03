import numpy as np
from random import shuffle

class KernelSVMClassifier(object):

  def __init__(self, X=None):
    self.Alpha = None
    self.kernel_matrix = None
    self.X = X
    self.sigma_square = 0.0 # bandwidth

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=2000, verbose=False):
    pass

  def RBF_kernel_svm_loss(self):
    """
    RBF kernel Soft-SVM loss function.
    """
    loss = 0.0

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
