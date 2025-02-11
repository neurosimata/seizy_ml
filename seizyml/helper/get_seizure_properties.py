# -*- coding: utf-8 -*-

### ------------------------------- Imports ------------------------------- ###
import os
import tqdm
import numpy as np
import pandas as pd
from seizyml.helper.event_match import get_szr_idx
### ----------------------------------------------------------------------- ###

def get_seizure_prop(parent_path, verified_predictions_dir, win):
    """
    Extracts seizure properties from verified prediction CSV files and saves the results to a new CSV.

    Parameters
    ----------
    parent_path : str
        Path to the parent directory containing verified predictions.
    verified_predictions_dir : str
        Subdirectory name with verified prediction CSV files.
    win : float
        Window size in seconds used to calculate seizure durations.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing seizure properties for each file.
    save_path : str
        File path to the saved CSV with seizure properties.
    """

    # get files
    ver_path = os.path.join(parent_path, verified_predictions_dir)
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))
    
    # get columns
    cols = ['seizure_number','avg_seizure_dur_sec', 'total_time_seizing_sec',
            'coefficient_of_variation', 'recording_dur_hrs']
    
    # save array
    save_array = np.zeros([len(filelist), len(cols)])
    for i in tqdm.tqdm(range(len(filelist))): # loop through csv files
        
        # load predictions and get seizure bounds
        file_path = os.path.join(ver_path, filelist[i])
        ver_pred = np.loadtxt(file_path, delimiter=',')
        idx_bounds = get_szr_idx(ver_pred)
        
        # extract properties
        if idx_bounds.shape[0] > 0:
            save_array[i,0] = idx_bounds.shape[0]                              # seizure number
            save_array[i,1] = (np.sum(ver_pred)*win)/idx_bounds.shape[0]       # average seizure duration (seconds)     
            save_array[i,2] = (save_array[i,0]* save_array[i,1]).sum()         # total time seizing (seconds)
            iei = np.diff(idx_bounds[:,0])
            save_array[i,3] = 100*np.std(iei)/np.mean(iei)                     # inter-event-inteval
        save_array[i,4] = ver_pred.shape[0] * win/3600
    
    # create dataframe and save
    df = pd.DataFrame(data = save_array, columns = cols)    
    df.insert(0,'file_id', filelist) 
    df['seizures_per_hour'] = df['seizure_number'] / df['recording_dur_hrs'] 
    df['percent_time_seizing'] = 100*(df['total_time_seizing_sec']/3600) / df['recording_dur_hrs']
    save_path = os.path.join(parent_path, 'seizure_properties.csv')
    df.to_csv(save_path, index=False)

    return df, save_path



