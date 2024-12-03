# -*- coding: utf-8 -*-         
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
from helper.io import load_data
from helper.event_match import get_szr_idx, clean_predictions
from helper.get_features import compute_selected_features
### ------------------------------------------------------------------------###


class ModelPredict:
    """
    Class for batch seizure prediction.

    Attributes
    ----------
    load_path : str
        The path to the directory containing the input data.
    save_path : str
        The path to the directory where the output data will be saved.
    channels : list
        A list of channel names.
    win : int
        The window size in seconds.
    fs : float
        The sampling frequency in Hz.
    min_seizure_duration : int
        The minimum duration of a seizure in seconds.
    erode : int
        The number of windows to erode from the start and end of a seizure.
    dilation : int
        The number of windows to dilate a seizure.

    Methods
    -------
    __init__(self, model_path: str, load_path: str, save_path: str, selected_features: list, channels: list, win: int, fs: float)
        Initializes the ModelPredict class.
    predict(self)
        Runs batch predictions.
    get_feature_pred(self, file_id)
        Gets predictions for a given file.
    save_idx(file_path, y_pred, bounds_pred)
        Saves user predictions to a CSV file as binary.
    """

    def __init__(self, model_path: str, load_path: str, save_path: str, 
                  channels: list, win: int, fs: float):
        """
        Initializes the ModelPredict class.

        Parameters
        ----------
        model_path : str
            The path to the trained model file.
        load_path : str
            The path to the directory containing the input data.
        save_path : str
            The path to the directory where the output data will be saved.
        channels : list
            A list of channel names.
        win : int
            The window size in seconds.
        fs : float
            The sampling frequency in Hz.

        Returns
        -------
        None
        """
        # Set the input parameters as class attributes
        self.load_path = load_path
        self.save_path = save_path
        self.channels = channels
        self.win = win
        self.fs = fs    
        # self.min_seizure_duration = 10 # minimum seizure duration
        # self.erode = int(self.min_seizure_duration/self.win)-1
        # self.dilation = 5 # in window bins
        
        # Load the trained model
        self.model = load(model_path +'.joblib')
        self.selected_features = self.model.feature_labels
        print('Model loaded:', self.model)

    def compute_metrics(self, file_id, y_pred, bounds_pred):
        """
        Computes and saves additional metrics such as the number of seizures detected 
        and the recording length for a given file to display during seizure verification.

        Parameters
        ----------
        file_id : str
            The file name with no extension.
        y_pred : 1D array
            The binary predictions.
        bounds_pred : 2D array
            The start and end points of detected seizures.

        Returns
        -------
        None
        """
        # Calculate the number of seizures detected
        num_seizures = bounds_pred.shape[0]
        
        # Calculate the recording length in hours
        recording_length = y_pred.shape[0] * self.win/60/60
        
        # Save these metrics in a dictionary
        metrics = {
            'file_id': file_id,
            'num_seizures': num_seizures,
            'recording_length': recording_length
        }

        # Write the metrics to a JSON file
        with open(os.path.join(self.save_path, file_id.replace('.h5','_metrics.json')), 'w') as f:
            json.dump(metrics, f)

    def predict(self):
        """
        Runs batch predictions.
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
            y_pred, bounds_pred = self.get_feature_pred(filelist[i])

            # Compute and save additional metrics
            self.compute_metrics(filelist[i], y_pred, bounds_pred)
            
            # Convert prediction to binary vector and save as .csv
            ModelPredict.save_idx(os.path.join(self.save_path, filelist[i].replace('.h5','.csv')), y_pred, bounds_pred)
            
        print('---> Predictions have been generated for: ', self.save_path + '.','\n')
        print('---------------------------------------------------------------------------\n')
            
               
    def get_feature_pred(self, file_id):
        """
        Gets predictions for a given file.

        Parameters
        ----------
        file_id : str
            The file name with no extension.

        Returns
        -------
        y_pred : 1D array
            The binary predictions.
        bounds_pred : 2D array
            The start and end points of detected seizures.
        """
        
        # get data and true labels
        data = load_data(os.path.join(self.load_path, file_id))
        
        # Eextract features and normalize
        features, _ = compute_selected_features(data, self.selected_features, self.channels, self.fs)
        features = StandardScaler().fit_transform(features)
        
        # get predictions
        y_pred = self.model.predict(features)
        clean_pred = clean_predictions(y_pred, operation='mdt')
        bounds_pred = get_szr_idx(clean_pred)
        
        return clean_pred, bounds_pred 

            
    def save_idx(file_path, y_pred, bounds_pred):
        """
        Saves user predictions to a CSV file as binary.

        Parameters
        ----------
        file_path : str
            The path to the file.
        y_pred : 1D array
            The binary predictions.
        bounds_pred : 2D array
            The start and end points of detected seizures.

        Returns
        -------
        None
        """
        # pre allocate file with zeros
        ver_pred = np.zeros(y_pred.shape[0])
        
        for i in range(bounds_pred.shape[0]):
            if bounds_pred[i,0] > 0:   
                ver_pred[bounds_pred[i,0]:bounds_pred[i,1]+1] = 1
                
        # save
        np.savetxt(file_path, ver_pred, delimiter=',', fmt='%i')

    
   
