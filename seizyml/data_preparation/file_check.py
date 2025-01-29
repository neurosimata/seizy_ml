# -*- coding: utf-8 -*-

### -------------------------------- IMPORTS ------------------------------ ###
import os
from tqdm import tqdm
from seizyml.helper.io import load_data
### ------------------------------------------------------------------------###

def check_main(parent_path, data_dir, processed_dir, model_predictions_dir):
    """
    Check the integrity of the main directories and their contents.

    Parameters
    ----------
    parent_path : str
        The parent directory containing the data, processed, and model predictions directories.
    data_dir : str
        The name of the data directory containing H5 data files.
    processed_dir : str
        The name of the processed directory.
    model_predictions_dir : str
        The name of the model predictions directory.

    Returns
    -------
    processed_check : bool
        True if the processed directory exists and has the same files as the data directory, False otherwise.
    model_predictions_check : bool
        True if the model predictions directory exists and has the same files as the processed directory, False otherwise.

    Notes
    -----
    This function checks the following:
    1. If the processed directory exists in the specified parent directory.
    2. If the model predictions directory exists in the specified parent directory.
    3. If the processed directory has the same files as the data directory (by comparing filenames without the '.h5' extension).
    4. If the model predictions directory has the same files as the processed directory (by comparing filenames without the '.csv' extension).

    Examples
    --------
    >>> parent_path = '/path/to/parent_directory/'
    >>> data_dir = 'h5_data'
    >>> processed_dir = 'processed_data'
    >>> model_predictions_dir = 'model_predictions'
    >>> processed_check, model_predictions_check = check_main(parent_path, data_dir, processed_dir, model_predictions_dir)
    >>> processed_check
    True
    >>> model_predictions_check
    True
    """
    
    # initiate check variables for processed and model predictions 
    processed_check = True
    model_predictions_check = True
    
    # get paths for h5 data, processed and model predictions
    h5_path = os.path.join(parent_path, data_dir)
    processed_path = os.path.join(parent_path, processed_dir)
    model_predictions_path = os.path.join(parent_path, model_predictions_dir)

    # check if paths exist    
    if not os.path.exists(processed_path):
        processed_check = False
    if not os.path.exists(model_predictions_path):
        model_predictions_check = False

    if processed_check:
        h5 = set(x.replace('.h5', '') for x in os.listdir(h5_path))
        processed = set(x.replace('.h5', '') for x in os.listdir(processed_path))
        if h5 != processed:
            processed_check = False
            
    if processed_check and model_predictions_check:
        model_predictions = {x.replace('.csv', '') for x in os.listdir(model_predictions_path) if x[-4:] == '.csv'}   
        if processed != model_predictions:
            model_predictions_check = False
        
    return processed_check, model_predictions_check

def check_verified(folder, data_dir, csv_dir):
    """
    Check if folders exist and if h5 files match csv files.

    Parameters
    ----------
    folder : dict, with config settings
    data_dir : str, data directory name
    true_dir : str, csv directory name

    Returns
    -------
    None,str, None if test passes, otherwise a string is returned with the name
    of the folder where the test did not pass

    """
    
    h5_path = os.path.join(folder, data_dir)
    ver_path = os.path.join(folder, csv_dir)
    if not os.path.exists(h5_path):
        return h5_path
    if not os.path.exists(ver_path):
        return ver_path
    h5 = {x.replace('.h5', '') for x in os.listdir(h5_path)}
    ver = {x.replace('.csv', '') for x in os.listdir(ver_path)}   
    if len(h5) != len(h5 & ver):
        return folder 

def check_h5_files(path, win, fs, channels=2):
    """
    Check H5 files for compatibility.

    Parameters
    ----------
    path : str
        The path to the directory containing the H5 data files.
    win : int or float
        The window size in seconds.
    fs : int or float
        The sampling frequency in Hz of the data.
    channels : int, optional
        The expected number of channels in the H5 files. The default is 2.

    Returns
    -------
    error : list
        Contains error messages.

    Notes
    -----
    This function checks each H5 file in the specified directory to ensure it meets the following criteria:
    1. The file has a '.h5' extension.
    2. The file can be successfully read and loaded as data.
    3. The number of channels in the data matches the specified 'channels'.
    4. The number of samples in each channel matches the expected length given 'win' and 'fs'.

    If any of the above conditions are not met, the function appends a descriptive error message
    to the 'error' list. Otherwise, the 'error' list remains empty, indicating all files passed
    the compatibility check.

    Examples
    --------
    >>> check_h5_files('/path/to/h5_files/', win=0.1, fs=1000, channels=2)
    ['File: file_name.h5 does not have the right dimensions. Got (500, 99, 3) instead of n:100:2']

    >>> check_h5_files('/path/to/h5_files/', win=0.2, fs=500, channels=3)
    ['File: another_file.h5 could not be read.']

    >>> check_h5_files('/path/to/h5_files/', win=0.5, fs=1000, channels=2)
    []
    """
    error = []
    # get file list 
    filelist = list(filter(lambda k: '.h5' in k, os.listdir(path)))
    if len(filelist) == 0:
        error.append('No H5 files were found.')
    
    for file in tqdm(filelist, total=len(filelist), desc='Checking files:'):
        try:
            data = load_data(os.path.join(path, file))
        except:
            error.append('File: ' + file + ' could not be read.')
        
        n_channels = data.shape[2]
        bin_size = data.shape[1]
        if (n_channels != channels) | (bin_size != int(fs*win)):
            error_str = 'File: ' + file + 'does not have the right dimensions. Got ' + str(data.shape) +' instead of n:'+ str(int(win*fs)) + ':' +str(channels)
            error.append(error_str)
    return error
            

























