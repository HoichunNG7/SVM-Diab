from svm.classifiers.linear_svm import linear_svm_loss_vectorized
import time
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

diabetes = pd.read_csv('svm/datasets/diabetes.csv')
x_train, x_test, y_train, y_test = train_test_split(np.array(diabetes.loc[:, diabetes.columns != 'Outcome']), np.array(diabetes['Outcome']), stratify=np.array(diabetes['Outcome']), random_state=66)

num_training = 480
num_validation = 96 
num_test = 160

# Change label set from {0, 1} to {-1, 1}
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = x_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = x_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = x_test[mask]
y_test = y_test[mask]

# Invoke sklearn tools
linear_svc = svm.SVC(kernel='linear')
rbf_svc = svm.SVC(kernel='rbf', probability=True)

# Model Selection
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}

# grid_search0 = GridSearchCV(linear_svc, param_grid)
# grid_search0.fit(X_train, y_train)
# best_parameters0 = grid_search0.best_estimator_.get_params()    
# print('linear hyperparameter C: ', best_parameters0['C'])
            
grid_search1 = GridSearchCV(rbf_svc, param_grid)
grid_search1.fit(X_train, y_train)
best_parameters1 = grid_search1.best_estimator_.get_params()    
print('RBF hyperparameter C: ', best_parameters1['C'])

print('\n')

# Training
linear_svc = svm.SVC(kernel='linear', C=1)
tic0 = time.time()
linear_svc.fit(X_train, y_train)
toc0 = time.time()
print('Linear svm training took %fs' % (toc0 - tic0))

rbf_svc = svm.SVC(kernel='rbf', C=best_parameters1['C'], probability=True)
tic1 = time.time()
rbf_svc.fit(X_train, y_train)
toc1 = time.time()
print('RBF kernel svm training took %fs' % (toc1 - tic1))
print('\n')

# Evaluation
y_train_pred = linear_svc.predict(X_train)
print('linear training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = linear_svc.predict(X_val)
print('linear validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
y_test_pred = linear_svc.predict(X_test)
print('linear test accuracy: %f' % (np.mean(y_test == y_test_pred), ))

y_train_pred = rbf_svc.predict(X_train)
print('RBF training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = rbf_svc.predict(X_val)
print('RBF validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
y_test_pred = rbf_svc.predict(X_test)
print('RBF test accuracy: %f' % (np.mean(y_test == y_test_pred), ))
print('\n')
