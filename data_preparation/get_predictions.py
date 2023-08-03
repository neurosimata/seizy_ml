# -*- coding: utf-8 -*-         
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
# User Defined
from helper.array_helper import find_szr_idx, merge_close
from helper import features
from helper.io_getfeatures import get_data, get_features_allch
### ------------------------------------------------------------------------###

# define model and features
model_name = 'gnb_model'
feature_idx = np.array([1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0])
param_list = (features.autocorr, features.line_length, features.rms, 
              features.mad, features.var, features.std, features.psd, 
              features.energy, features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr, features.signal_covar,
                       features.signal_abs_covar,)


class ModelPredict:
    """
    Class for batch seizure prediction.
    """
    
    def __init__(self, load_path, save_path, win, fs):
        """

        Parameters
        ----------
        load_path : str
        save_path : str
        win : int
        fs : int/float

        Returns
        -------
        None.

        """
            
        # get main path and load properties file
        self.load_path = load_path
        self.save_path = save_path
        self.win = win
        self.fs = fs    
        self.erode = 1 # in window bins
        self.merge = 5 # in window bins

        # load model
        self.feature_select = np.where(feature_idx)[0]
        self.model = load(model_name +'.joblib')
        print('Model loaded:', self.model)


    def predict(self):
        """
        Run batch predictions.
        """
       
        print('---------------------------------------------------------------------------\n')
        print('---> Initiating Predictions.', '\n')
       
        # Create path prediction path
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        
        # Get file list
        filelist = list(filter(lambda k: '.h5' in k, os.listdir(self.load_path)))
        
        # loop files (multilple channels per file)
        for i in tqdm(range(len(filelist)), desc = 'Progress'):
            
            # Get predictions (1D-array)
            data, bounds_pred = self.get_feature_pred(filelist[i])
            
            # Convert prediction to binary vector and save as .csv
            ModelPredict.save_idx(os.path.join(self.save_path, filelist[i].replace('.h5','.csv')), data, bounds_pred)
            
        print('---> Predictions have been generated for: ', self.save_path + '.','\n')
        print('---------------------------------------------------------------------------\n')
            
               
    def get_feature_pred(self, file_id):
        """
        Get predictions

        Parameters
        ----------
        file_id : str, file name with no extension

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        bounds_pred : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        # get data and true labels
        data = get_data(os.path.join(self.load_path, file_id))
        
        # Eextract features and normalize
        x_data, labels = get_features_allch(data, param_list, cross_ch_param_list)
        x_data = StandardScaler().fit_transform(x_data)
        
        # get predictions
        y_pred = self.model.predict(x_data[:,self.feature_select])
        bounds_pred = find_szr_idx(y_pred, dur=self.erode)
        
        # if seizures are detected, merge close segments
        if bounds_pred.shape[0] > 0:
            bounds_pred = merge_close(bounds_pred, merge_margin=self.merge)
            
        return data, bounds_pred 

            
    def save_idx(file_path, data, bounds_pred):
        """
        Save user predictions to csv file as binary
    
        Parameters
        ----------
        file_path : Str, path to file save
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        bounds_pred : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures) 
        
        Returns
        -------
        None.
    
        """
        # pre allocate file with zeros
        ver_pred = np.zeros(data.shape[0])
    
        for i in range(bounds_pred.shape[0]):   # assign index to 1
        
            if bounds_pred[i,0] > 0:   
                ver_pred[bounds_pred[i,0]:bounds_pred[i,1]+1] = 1
            
        # save file
        np.savetxt(file_path, ver_pred, delimiter=',', fmt='%i')

    
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            