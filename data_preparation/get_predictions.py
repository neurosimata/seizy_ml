# -*- coding: utf-8 -*-         
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
# User Defined
from helper.array_helper import find_szr_idx, merge_close
from helper import features
from helper.io_getfeatures import get_data, get_features_allch
### ------------------------------------------------------------------------###
              
class ModelPredict:
    """
    Class for batch seizure prediction.
    """
    
    def __init__(self, property_dict):
        """

        Parameters
        ----------
        property_dict : Dict, contains essential info for data load and processing

        """
        
        # get main path and load properties file
        self.gen_path = property_dict['main_path']
        self.filt_dir = property_dict['filt_dir']
        
        # create raw pred path
        self.rawpred_dir = os.path.join(self.gen_path, property_dict['rawpred_dir'])
        
        # get sampling rate
        self.fs = property_dict['new_fs']
        self.win = property_dict['win']

        # read method parameters into dataframe
        df = pd.read_csv(os.path.join('train','selected_method.csv'))
        self.thresh = np.array(df.loc[0][df.columns.str.contains('Thresh')])
        self.weights = np.array(df.loc[0][df.columns.str.contains('Weight')])
        self.enabled = np.array(df.loc[0][df.columns.str.contains('Enabled')])
        
        # get feature names
        self.feature_names = df.columns[df.columns.str.contains('Enabled')]
        self.feature_names = np.array([x.replace('Enabled_', '') for x in  self.feature_names])


    def predict(self):
        """
        Run batch predictions.
        """
       
        print('---------------------------------------------------------------------------\n')
        print('---> Initiating Predictions for:', self.rawpred_dir + '.', '\n')
       
        # Create path prediction path
        if os.path.exists(self.rawpred_dir) is False:
            os.mkdir(self.rawpred_dir)
        
        # Get file list
        filelist = list(filter(lambda k: '.h5' in k, os.listdir(os.path.join(self.gen_path, self.filt_dir))))
        
        # loop files (multilple channels per file)
        for i in tqdm(range(len(filelist)), desc = 'Progress'):
            
            # Get predictions (1D-array)
            data, bounds_pred = self.get_feature_pred(filelist[i].replace('.h5',''))
            
            # Convert prediction to binary vector and save as .csv
            ModelPredict.save_idx(os.path.join(self.rawpred_dir, filelist[i].replace('.h5','.csv')), data, bounds_pred)
            
        print('---> Predictions have been generated for: ', self.rawpred_dir + '.','\n')
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
        
        # define parameter list
        param_list = (features.autocorr, features.line_length, features.rms, 
                      features.mad, features.var, features.std, features.psd, 
                      features.energy, features.get_envelope_max_diff,)
        cross_ch_param_list = (features.cross_corr, features.signal_covar,
                               features.signal_abs_covar,)
        
        # get data and true labels
        data = get_data(self.gen_path, file_id,
                        inner_path={'data_path':self.filt_dir},
                        load_y=False)
        
        # Eextract features and normalize
        x_data, labels = get_features_allch(data, param_list, cross_ch_param_list)
        x_data = StandardScaler().fit_transform(x_data)
        
        # get predictions
        thresh = (np.mean(x_data) + self.thresh * np.std(x_data))               # get threshold vector
        y_pred_array = (x_data > thresh)                                        # get predictions for all conditions
        y_pred = y_pred_array * self.weights * self.enabled                     # get predictions based on weights and selected features
        y_pred = np.sum(y_pred, axis=1) / np.sum(self.weights * self.enabled)   # normalize to weights and selected features
        y_pred = y_pred > 0.5                                                   # get popular vote
        bounds_pred = find_szr_idx(y_pred, dur=1)                               # get predicted seizure index
        
        # if seizures are detected, merge close segments
        if bounds_pred.shape[0] > 0:
            bounds_pred = merge_close(bounds_pred, merge_margin=5)
            
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

    
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            