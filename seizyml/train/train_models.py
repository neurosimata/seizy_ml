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
    Return the indices of selected features within the full set of features.
    
    Parameters
    ----------
    selected_features : list of str
        A list of feature names to locate within the complete feature set.
    all_features : numpy.ndarray
        A 1D array containing all available feature names.
        
    Returns
    -------
    numpy.ndarray
        An array of indices corresponding to the positions of each feature in 
        `selected_features` found in `all_features`.
        
    Raises
    ------
    IndexError
        If a feature in `selected_features` is not found in `all_features`.
    """

    # Loop through each selected feature to find its index in the all_features array
    indices = []
    for feature in selected_features:
        index = np.where(all_features == feature)[0][0]
        indices.append(index)

    return np.array(indices)

def grid_train(model_path, model_name, x, y, selected_features, feature_labels):
    """
    Perform hyperparameter tuning via grid search for a specified model on multiple feature sets,
    saving the best model based on F1 score per feature set.
    
    Parameters
    ----------
    model_path : str
        Directory path where the best model files will be saved.
    model_name : str
        The key name of the machine learning model in the `models` dictionary.
    x : numpy.ndarray
        A 2D array with shape (n_samples, n_features) containing the feature data.
    y : numpy.ndarray
        A 1D array containing the classification target labels.
    selected_features : dict
        A dictionary where each key is a feature set name and each value is a list of feature names
        (selected from `feature_labels`) to be used for that feature set.
    feature_labels : numpy.ndarray
        A 1D array of feature names corresponding to the columns in `x`.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the grid search results with columns including:
        ['ID', 'HYPERPARAMETERS', ...], where additional columns represent performance metrics
        defined in the `metrics` dictionary.
    
    Notes
    -----
    - The best estimator for each feature set is saved as a joblib file using a unique ID.
    - The best index is computed using a custom transformation on the ranking of the refit metric.
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
    Train a collection of machine learning models using grid search over multiple feature sets,
    save the trained models, and return a summary DataFrame of the results.
    
    Parameters
    ----------
    trained_model_path : str
        Directory path where the trained model files will be saved. The directory will be created if it does not exist.
    x : numpy.ndarray or pandas.DataFrame
        A 2D array or DataFrame containing the feature data.
    y : numpy.ndarray or pandas.Series
        A 1D array or Series containing the target labels.
    selected_features : dict
        A dictionary mapping each feature set name to a list of selected feature names.
    feature_labels : numpy.ndarray
        A 1D array of feature names corresponding to the columns in `x`.
    
    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame summarizing the grid search results for all models,
        including unique model IDs, hyperparameters, and performance metrics, along with a 'model' column.
    
    Notes
    -----
    - This function iterates over all models defined in the `models` dictionary.
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
    Train a new seizure detection model using labeled training data.
    Depending on the `process` parameter, this function can compute features from raw data,
    train the model using grid search, or perform both steps sequentially.
    
    Parameters
    ----------
    usr : dict
        Dictionary of user settings that includes model parameters, feature extraction options,
        and data processing parameters.
    train_path : str
        Path to the training data directory containing raw data files ('.h5') and corresponding label files ('.csv').
    process : str or None
        Specifies the process to execute:
          - 'compute_features': Compute features from raw data.
          - 'train_model': Train the model using precomputed features.
          - Any other value or None: Execute both feature computation and model training.
    
    Returns
    -------
    str
        The file path to the saved best trained model (joblib file).
    
    Raises
    ------
    ValueError
        If the training data files are missing or if the file structure validation fails.
    
    Notes
    -----
    - When computing features, the raw data is preprocessed, features are computed and standardized,
      and then saved to disk.
    - Grid search is used for hyperparameter tuning, and the best model is selected based on the F1 score.
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
        train_df.to_csv(os.path.join(trained_model_path, 'trained_models.csv'), index=False)
        best_model_id = train_df.loc[train_df['F1'].idxmax(), 'ID']
        model_path = os.path.join(trained_model_path, f'{best_model_id}.joblib')

        return model_path

