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

def train_model(usr, train_path, process):
    """
    Train a new seizure detection model using labeled data.

    Parameters:
    ----------
    usr : dict
        User settings containing model and data processing parameters.
    train_path : str
        Path to the training data directory containing `.h5` files and corresponding `.csv` labels.
    process : str
        Specifies which process to run:
        - 'compute_features': Computes features from training data.
        - 'train_model': Trains the model using existing features.
        - None: Runs both feature computation and model training.

    Returns:
    -------
    str
        Path to the saved trained model.

    Raises:
    ------
    ValueError
        If the provided process type is invalid.
    FileNotFoundError
        If training data or label files are missing.
    """

    # imports
    import click
    from seizyml.train.train_models import train_and_save_models
    from seizyml.helper.io import load_data, save_data
    from seizyml.data_preparation.preprocess import PreProcess
    from sklearn.preprocessing import StandardScaler
    from seizyml.helper.get_features import compute_features
    from seizyml.train.select_features import select_features
    from tqdm import tqdm

    # Define process types
    process_type_options = ['compute_features', 'train_model']
    process_types = [process] if process in process_type_options else process_type_options

    if 'compute_features' in process_types:
        label_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]
        h5_files = [f.replace('.csv', '.h5') for f in label_files]

        # Check data consistency
        from seizyml.data_preparation.file_check import train_file_check
        train_file_check(train_path, h5_files, label_files, usr['win'], usr['fs'], usr['channels'])

        # Compute features
        click.secho("⚙️ Computing features...", fg='green')
        x_all, y_all = [], []
        for h5_file, label_file in tqdm(zip(h5_files, label_files), total=len(h5_files)):
            x = load_data(os.path.join(train_path, h5_file))
            obj = PreProcess(None,None, fs=usr['fs'])
            x_clean = obj.filter_clean(x)
            features, feature_labels = compute_features(x_clean, usr['features'], usr['channels'], usr['fs'])
            features = StandardScaler().fit_transform(features)
            x_all.append(features)
            y_all.append(np.loadtxt(os.path.join(train_path, label_file)))

        # Save computed features
        save_data(os.path.join(train_path, 'features.h5'), np.concatenate(x_all))
        save_data(os.path.join(train_path, 'y.h5'), np.concatenate(y_all))
        np.savetxt(os.path.join(train_path, 'feature_labels.txt'), feature_labels, fmt='%s')

    if 'train_model' in process_types:
        click.secho("🎯 Training model...", fg='green')

        # load features, ground truth labels, and feature labels
        features = load_data(os.path.join(train_path, 'features.h5'))
        y = load_data(os.path.join(train_path, 'y.h5'))
        feature_labels = np.loadtxt(os.path.join(train_path, 'feature_labels.txt'), dtype=str)

        # select feature sets
        selected_features = select_features(features, y, feature_labels,
                                            r_threshold=usr['feature_select_thresh'],
                                            feature_size=usr['feature_size'],
                                            nleast_correlated=usr['nleast_corr'])

        # train models and find the one with best f1 score
        trained_model_path = os.path.join(train_path, usr['trained_model_dir'])
        train_df = train_and_save_models(trained_model_path, features, y, selected_features, feature_labels)
        best_model_id = train_df.loc[train_df['F1'].idxmax(), 'ID']
        model_path = os.path.join(trained_model_path, f'{best_model_id}.joblib')

        return model_path

