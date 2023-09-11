# -*- coding: utf-8 -*-

### ------------------------------- IMPORTS ----------------------------- ###
import numpy as np
from sklearn.naive_bayes import GaussianNB
### --------------------------------------------------------------------- ###


# define metrics and Kfold partitions
fit_metric = 'BALANCED_ACCURACY'
best_metric_rank = 'rank_test_' + fit_metric
metrics = {'AUC':'roc_auc', 'BALANCED_ACCURACY':'balanced_accuracy', 'PRECISION':'precision', 'RECALL':'recall', 'F1':'f1'}
grid_metrics = 'mean_test_'


# define models and hyperparameters
models = {  
            'gaussian_nb' : GaussianNB(),       
          }

hyper_params = {
                'gaussian_nb': {
                                'var_smoothing': np.logspace(-2,-8, num=7)},
                }