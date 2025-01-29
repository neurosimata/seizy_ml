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
from joblib import dump
from seizyml.train.model_settings import models, hyper_params, metrics, fit_metric
njobs = int(multiprocessing.cpu_count()*.8)
### --------------------------------------------------------------------- ###

def get_feature_indices(selected_features, all_features):
    """
    Returns the indices in the X matrix corresponding to the selected features.

    Parameters:
    - selected_features (list): A list of selected feature names.
    - all_features (np.array): A numpy array of all feature names.

    Returns:
    - indices (np.array): An array of indices corresponding to the selected features in the X matrix.
    """

    # Loop through each selected feature to find its index in the all_features array
    indices = []
    for feature in selected_features:
        index = np.where(all_features == feature)[0][0]
        indices.append(index)

    return np.array(indices)

def grid_train(model_path, model_name, x, y, selected_features, feature_labels):
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
    selected_features : dict
        Keys are feature-set names and values are list with names of selected features.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing unique identifiers for each model, the best hyperparameters, and performance metrics.
    """

    # define Kfold 
    cv = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    
    # iterate through feature sets
    data_list = []
    for feature_set in tqdm(selected_features, total=len(selected_features)):
        
        # perform grid search for particular feature set
        search = GridSearchCV(estimator=models[model_name],
                              param_grid=hyper_params[model_name],
                              n_jobs=njobs,
                              cv=cv,
                              verbose=0,
                              scoring=metrics,
                              refit=fit_metric)
        # get feature index
        sel_feature_idx = get_feature_indices(selected_features[feature_set], feature_labels)
        search.fit(x[:, sel_feature_idx], y)
        
        # create unique ID and save model with features labels
        uid = uuid4().hex
        best_model = search.best_estimator_
        best_model.feature_labels = selected_features[feature_set]
        dump(best_model, os.path.join(model_path, uid + '.joblib'))

        # get metrics and best parameters
        row = [uid, search.best_params_]
        best_metric_rank = 'rank_test_' + fit_metric
        results = search.cv_results_
        best_idx = np.argmin(results[best_metric_rank]-1)
        for m in metrics:
            scores = results['mean_test_' + m]
            row.append(scores[best_idx])
        data_list.append(row)
        
        df = pd.DataFrame(data=data_list, columns=['ID', 'HYPERPARAMETERS']+ list(metrics.keys()))
    return df

def train_and_save_models(trained_model_path, x, y, selected_features, feature_labels):
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
        print('Training model: ', m)
        train_df = grid_train(trained_model_path, m,  x, y, selected_features, feature_labels)
        train_df['model'] = m
        df_list.append(train_df)
   
    return pd.concat(df_list).reset_index(drop=True)