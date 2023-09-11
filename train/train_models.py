# -*- coding: utf-8 -*-

### ------------------------------- IMPORTS ----------------------------- ###
import os
import itertools
import numpy as np
import pandas as pd
from uuid import uuid4
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from model_settings import models, hyper_params
from sklearnex import patch_sklearn 
patch_sklearn(verbose=True)
from joblib import dump, Parallel, delayed
from helper.progressbar import tqdm_joblib
njobs = int(multiprocessing.cpu_count()*.8)
### --------------------------------------------------------------------- ###

def grid_train(x, y, model_name, feature_space, fit_metric='balanced_accuracy'):
    """
    Find best hyperparameters for each model using gridsearch.

    Parameters
    ----------
    x : 2d array, feature data (time bins, features)
    y : 1d array, classification
    model_name : str,
    feature_space : pandas Df, with feature index

    Returns
    -------
    best_hyperparameters : pd.DataFrame, hyperparameters for each feature set

    """

    # define Kfold 
    cv = StratifiedKFold(n_splits=4, random_state=2, shuffle=True)
    
    best_hyperparameters = []
    for i in tqdm(range(len(feature_space))):
        # perform grid search for particular feature set
        search = GridSearchCV(estimator=models[model_name],
                              param_grid=hyper_params[model_name],
                              n_jobs=njobs,
                              cv=cv,
                              verbose=3,
                              refit=fit_metric)
        feature_select = np.where(feature_space.loc[i])[0]
        search.fit(x[:, feature_select], y)
        
        # get best hyper parameters
        best_hyperparameters.append(search.best_params_)

    return pd.DataFrame(best_hyperparameters)

def train_and_save_models(trained_model_path, x, y, feature_space, train_process='parallel'):
    """
    Train selected machine learning models, save them, and record them in a DataFrame.
    
    Parameters
    ----------
    trained_model_path: str, Path to save the trained model.
    x : array-like, Features of the training data.
    y : array-like, Target labels of the training data.
    feature_space : pd.DataFrame, DataFrame containing feature space information.
    train_process: str, "serial" or "parallel" processing
    
    Returns
    -------
    train_df : pd.DataFrame, DataFrame containing unique identifiers and properties for each trained model.
    """
    
    # create training dir
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    
    # perform grid train and get best model hyperparameters
    model_hps = {}
    print('-> Looking for best hyperparameters:')
    for m in models:
        best_hyperparams = grid_train(x, y, m, feature_space)
        model_hps.update({m:best_hyperparams})
    
    # create dataframe for each model, feature space
    df_data = list(itertools.product(model_hps.keys(), np.arange(len(feature_space))))
    train_df = pd.DataFrame(columns=['model', 'feature_space_id'], data=df_data)
    
    # add hyperparameters to train df
    train_df['hyperparameters'] = np.nan
    train_df['hyperparameters'] = train_df['hyperparameters'].astype('object')
    for idx, row in train_df.iterrows():
        model_name = row['model']
        feature_space_id = row['feature_space_id']
        hyperparameters = model_hps[model_name].iloc[feature_space_id].to_dict()
        train_df.at[idx, 'hyperparameters'] = hyperparameters

    # train models
    train_ml = TrainModels(model_path=trained_model_path, 
                           train_df=train_df,
                           x_train=x, 
                           y_train=y,
                           feature_space=feature_space)
    train_df = train_ml.train_models(process=train_process)
    
    return train_df


class TrainModels:
    """
    Handles the training, saving, and identifier assignment of machine learning models based on configurable settings.
    
    Methods
    -------
    - __init__ : Initializes the class with provided data and configuration settings.
    
    - train_model : Trains a single model based on selected features and hyperparameters.
    
    - train_serial : Trains and saves models sequentially.
    
    - train_parallel : Similar to train_serial, but performs training in parallel for speed-up.
    
    - train_models : Entry point for training models, supports both 'serial' and 'parallel' modes. 
                     Returns a DataFrame with unique identifiers added.
    """
    
    def __init__(self, model_path, train_df, x_train, y_train, feature_space):
        """
        Initialize the model with provided data and configuration.
    
        Parameters
        ----------
        model_path : str, Path to save the trained model.
        train_df : pd.DataFrame, The DataFrame containing training data.
        x_train : array-like, Features of the training data.
        y_train : array-like, Target labels of the training data.
        feature_space : list, Selected feature names.
    
        Returns
        -------
        None
        """
        
        self.model_path = model_path
        self.train_df = train_df
        self.x_train = x_train
        self.y_train = y_train
        self.feature_space = feature_space
        
    def train_model(self, idx, row):
        """
        Train model based on selected features and parameters.
    
        Parameters
        ----------
        idx: int, pandas index
        row : pd.series, model hyperparameters
        
        Returns
        -------
        model : obj: trained model
        idx: int, pandas index
        uid : str, unique id for model
        """

        # generate unique ID
        uid = uuid4().hex
    
        # get features
        feature_id = row.feature_space_id
        feature_select = np.where(self.feature_space.loc[feature_id])[0]
        
        # get hyperparameters
        prms = row.hyperparameters
        prms = {k: (None if v == '' else v) for k, v in prms.items()}
    
        # train model
        model = models[row.model]
        model.set_params(**prms)
        model.fit(self.x_train[: feature_select], self.y_train)
    
        return model, idx, uid
    
    def train_serial(self):
        """
        Train and saves models in a loop.
    
        Returns
        -------
        lists : list of tuples, train models, train df idx, uids
        """
        
        lists = []
        for idx, row in tqdm(self.train_df.iterrows(), total=len(self.train_df)):
           model, idx, uid = self.train_model(idx, row)
           lists.append((model, idx, uid))

        return  lists

    def train_parallel(self):
        """
        Train and saves models in parallel.
    
        Returns
        -------
        lists : list of tuples, train models, train df idx, uids
    
        """

        par = Parallel(n_jobs=njobs, backend='loky')
        with tqdm_joblib(tqdm(desc='Progress:', total=len(self.train_df))) as progress_bar:  # NOQA
            lists = par(delayed(self.train_model)(idx, row) for idx, row in self.train_df.iterrows())
    
        return lists
    
    def train_models(self, process='serial'):
        """
        Test models.
        
        Parameters
        ----------
        process : str, the default is 'serial'.

        Returns
        -------
        train_df_uid : pandas df, train df with uids

        """
        
        # train and save models
        print('\n----> Training models....', process, 'mode.\n')
        if process == 'serial':
            lists = self.train_serial()
        elif process == 'parallel':
            lists = self.train_parallel()  
        
        model_list, idx_list, uid_list = zip(*lists)
        
        # save models
        for (trained_model, uid) in zip(model_list, uid_list):
                dump(trained_model, os.path.join(self.model_path, uid + '.joblib'))
                
        # add ids to dataframe
        train_df_uid = self.train_df.copy()
        train_df_uid.loc[idx_list,'id'] = uid_list
        train_df_uid = train_df_uid[['id', 'model', 'feature_space_id', 'hyperparameters']]
        
        return train_df_uid


