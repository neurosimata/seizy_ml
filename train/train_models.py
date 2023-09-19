# -*- coding: utf-8 -*-

### ------------------------------- IMPORTS ----------------------------- ###
import os
import numpy as np
import pandas as pd
from uuid import uuid4
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn 
patch_sklearn(verbose=True)
from joblib import dump
from train.model_settings import models, hyper_params, metrics, fit_metric
njobs = int(multiprocessing.cpu_count()*.8)
### --------------------------------------------------------------------- ###

def grid_train(model_path, model_name,  x, y, feature_space):
    """
    Perform hyperparameter tuning for a specific machine learning model on various feature sets.
    Save the best models and return a DataFrame summarizing the tuning results.

    Parameters
    ----------
    model_path : str
        Directory path where the best model files will be saved.
    model_name : str
        Name of the machine learning model as defined in the 'models' dictionary.
    x : np.ndarray
        2D array containing feature data with dimensions (time bins, features).
    y : np.ndarray
        1D array containing the classification target labels.
    feature_space : pd.DataFrame
        DataFrame specifying which features to use for each tuning run. Each row represents a feature set.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing unique identifiers for each model, the best hyperparameters, and performance metrics.
    """

    # define Kfold 
    cv = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    
    data_list = []
    for i in tqdm(range(len(feature_space))):
        # perform grid search for particular feature set
        search = GridSearchCV(estimator=models[model_name],
                              param_grid=hyper_params[model_name],
                              n_jobs=njobs,
                              cv=cv,
                              verbose=3,
                              scoring=metrics,
                              refit=fit_metric)
        feature_select = np.where(feature_space.loc[i])[0]
        search.fit(x[:, feature_select], y)
        
        # create unique ID and save model
        uid = uuid4().hex
        best_model = search.best_estimator_
        dump(best_model, os.path.join(model_path, uid + '.joblib'))
        
        # create row list
        row = [uid, search.best_params_]
        
        # get metrics
        best_metric_rank = 'rank_test_' + fit_metric
        results = search.cv_results_
        best_idx = np.argmin(results[best_metric_rank]-1)
        for m in metrics:
            scores = results['mean_test_' + m]
            row.append(scores[best_idx])
        data_list.append(row)
        
        df = pd.DataFrame(data=data_list, columns=['ID', 'HYPERPARAMETERS']+ list(metrics.keys()))
    return df


def train_and_save_models(trained_model_path, x, y, feature_space):
    """
    Train a selection of machine learning models on a given dataset, saving each trained model and summarizing 
    the results in a DataFrame.

    Parameters
    ----------
    trained_model_path : str
        Directory path where the trained models will be saved.
    x : np.ndarray or pd.DataFrame
        2D array or DataFrame containing feature data.
    y : np.ndarray or pd.Series
        1D array or Series containing the classification target labels.
    feature_space : pd.DataFrame
        DataFrame specifying which features to use for each model. Each row represents a feature set.

    Returns
    -------
    train_df : pd.DataFrame
        DataFrame containing unique identifiers for each model, along with hyperparameters and performance metrics.
    """
    
    # create training dir
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    
    # perform grid train, save best estimators
    df_list = []
    for m in models:
        train_df = grid_train(trained_model_path, m,  x, y, feature_space)
        df_list.append(train_df)
   
    return pd.concat(df_list).reset_index(drop=True)