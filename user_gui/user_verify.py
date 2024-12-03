# -*- coding: utf-8 -*-

### ---------------- IMPORTS ------------------ ###
import os
import json
import tables
from pick import pick
import numpy as np
# User Defined
from helper.event_match import get_szr_idx
### ------------------------------------------- ###


class UserVerify:
    """
        Class for user verification of detected seizures.
    """
    
    # class constructor (data retrieval)
    def __init__(self, parent_path, processed_dir, model_predictions,
                 verified_predictions_dir):
        """
        
        Parameters
        ----------
        parent_path : str
        processed_dir : str
        model_predictions : str
        verified_predictions_dir : str

        Returns
        -------
        None.

        """

        # set full paths 
        self.processed_path = os.path.join(parent_path, processed_dir)
        self.model_predictions_path = os.path.join(parent_path, model_predictions)
        self.verified_predictions_path = os.path.join(parent_path, verified_predictions_dir)

        # make path if it doesn't exist
        if os.path.exists(self.verified_predictions_path) is False:
            os.mkdir(self.verified_predictions_path)

    def read_metrics(self, file_id):
        """
        Read metrics from json file.
        
        Parameters
        ----------
        file_id : str
            file id of file to read metrics from.
            
        Returns
        -------
        metrics : dict
            dictionary of metrics.
                
        """
        json_path = os.path.join(self.model_predictions_path, f"{file_id}_metrics.json")
        try:
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                print(metrics)
        except FileNotFoundError:
            metrics = {'file_id': file_id, 'num_seizures': 'N/A', 'recording_length': 'N/A'}
        return metrics
        
    def select_file(self):
        """
        Select file to load from list. Adds stars next to files that have been scored already.
        
        Returns
        -------
        option : Str, selection of file id
        """
       
        # get all files in raw predictions folder 
        rawpredlist = list(filter(lambda k: '.csv' in k, os.listdir(self.model_predictions_path)))
       
        # get all files in user verified predictions
        verpredlist = list(filter(lambda k: '.csv' in k, os.listdir(self.verified_predictions_path)))
       
        # get unique list
        not_analyzed_filelist = list(set(rawpredlist) - set(verpredlist))
        
        # remaining filelist
        analyzed_filelist = list(set(rawpredlist) - set(not_analyzed_filelist))
        
        # filelist
        filelist = not_analyzed_filelist + analyzed_filelist

        # create display list
        display_list = []
        lists = [not_analyzed_filelist, analyzed_filelist]
        append_strs = ['','***']
        for append_str, filelists in zip(append_strs, lists):
            for file in filelists:
                metrics = self.read_metrics(file.replace('.csv',''))
                display_list.append(append_str + ' ' + str(metrics)[1:-1].replace("'",""))
        
        # select from command list
        title = 'Please select file for analysis: '
        display_list, index = pick(display_list, title, indicator = '-> ')

        return filelist[index]


    def get_bounds(self, file_id):
        """
        Load data and calulate seizure bounds from predictions.

        Parameters
        ----------
        file_id : String

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        idx_bounds : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        print('-> File being analyzed: ', file_id)

        # Get predictions
        pred_path = os.path.join(self.model_predictions_path, file_id)
        bin_pred = np.loadtxt(pred_path, delimiter=',', skiprows=0)
        idx_bounds = get_szr_idx(bin_pred)
        
        # load raw data for visualization
        data_path = os.path.join(self.processed_path, file_id.replace('.csv','.h5'))
        f = tables.open_file(data_path, mode='r')
        data = f.root.data[:]
        f.close()
        
        # check whether to continue
        print('>>>>', idx_bounds.shape[0], 'seizures detected')
        
        return data, idx_bounds


              
    def save_emptyidx(self, data_len, file_id):
         """
         Save user predictions to csv file as binary
        
         Returns
         -------
         None.
        
         """
         # pre allocate file with zeros
         ver_pred = np.zeros(data_len)
         
         # save file
         np.savetxt(os.path.join(self.verified_predictions_path, file_id), ver_pred, delimiter=',',fmt='%i')
         print('Verified predictions for ', file_id, ' were saved\n')
    
    

       