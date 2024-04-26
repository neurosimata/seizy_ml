# -*- coding: utf-8 -*-

### ------------------------------- IMPORTS ----------------------------- ###
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
### --------------------------------------------------------------------- ###


# define metrics and Kfold partitions
fit_metric = 'BALANCED_ACCURACY'
metrics = {'AUC':'roc_auc', 'BALANCED_ACCURACY':'balanced_accuracy', 'PRECISION':'precision', 'RECALL':'recall', 'F1':'f1'}

# define models and hyperparameters
models = {  
            'gaussian_nb' : GaussianNB(),
            'sgd': SGDClassifier(),            
          }

hyper_params = {
                
                'gaussian_nb': {
                                'var_smoothing': np.logspace(-2,-8, num=7)},

                'sgd' :{
                          'alpha': [0.000001, 0.0001, 0.001, 0.01,],
                          'max_iter': [1000,],
                          'learning_rate': ['adaptive'],
                          'penalty': ['l2', 'l1', None],
                          'early_stopping': [False],
                          'eta0': [0.001 ,0.01],
                          'tol' : [1e-3, 1e-4],
                          'class_weight': ['balanced'],
                          'loss': ['log_loss'],
                          'validation_fraction': [0.2],
                      },

                }