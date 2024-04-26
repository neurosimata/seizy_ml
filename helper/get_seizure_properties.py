# -*- coding: utf-8 -*-

### ------------------------------- Imports ------------------------------- ###
import os
import tqdm
import numpy as np
import pandas as pd
from helper.array_helper import find_szr_idx
### ----------------------------------------------------------------------- ###

def get_seizure_prop(parent_path, verified_predictions_dir, win):
    """
    Get seizure properties and save to csv file.

    Parameters
    ----------
    parent_path
    verified_predictions_dir
    win
    

    Returns
    -------
    df : pd.DataFrame, containg seizure properties per verified csv
    save_path: str,
    """

    # get properties 
    ver_path = os.path.join(parent_path, verified_predictions_dir)
    
    # get csv file list
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))
    
    # get columns
    cols = ['seizure_number','avg_seizure_dur_sec', 'total_time_seizing_sec',
            'recording_dur_hrs']
    
    # save array
    save_array = np.zeros([len(filelist), len(cols)])
    
    for i in tqdm.tqdm(range(len(filelist))): # loop through csv files
        
        # get path csv file containing predicitons (one channel per column)
        file_path = os.path.join(ver_path, filelist[i])
        
        # load csv
        ver_pred = np.loadtxt(file_path, delimiter=',')

        # get seizure segments
        idx_bounds = find_szr_idx(ver_pred, dur=1)
        
        # save to array
        if idx_bounds.shape[0]>0:
            save_array[i,0] = idx_bounds.shape[0]                              # seizure number
            save_array[i,1] = (np.sum(ver_pred)*win)/idx_bounds.shape[0]       # average seizure duration (seconds)     
            save_array[i,2] = (save_array[i,0]* save_array[i,1]).sum()         # total time seizing (seconds)
        
        save_array[i,3] = ver_pred.shape[0] * win/3600
    
    # create dataframe
    df = pd.DataFrame(data = save_array, columns = cols)    
    df.insert(0,'file_id', filelist) # append file id
    
    # get seizure frequency and time seizing per hour
    df['seizures_per_hour'] = df['seizure_number'] / df['recording_dur_hrs'] 
    df['percent_time_seizing'] = 100*(df['total_time_seizing_sec']/3600) / df['recording_dur_hrs']
    
    # save file
    save_path = os.path.join(parent_path, 'seizure_properties.csv')
    df.to_csv(save_path, index=False)

    return df, save_path



