# -*- coding: utf-8 -*-

### -------------------------------- IMPORTS ------------------------------ ###
import os
from tqdm import tqdm
from seizyml.helper.io import load_data
### ------------------------------------------------------------------------###

def train_file_check(train_path, h5_files, label_files, win, fs, channels):
    """
    Check the existence and structure of H5 and label CSV files.

    Parameters
    ----------
    train_path : str
        Path to the directory containing training files.
    h5_files : list
        List of H5 file names.
    label_files : list
        List of corresponding CSV label file names.
    win : int or float
        The window size in seconds.
    fs : int or float
        The sampling frequency in Hz of the data.
    channels : int, optional
        The expected number of channels in the H5 files. The default is 2.

    Raises
    ------
    ValueError
        If any file is missing or improperly structured.
    """
    for x_path, y_path in zip(h5_files, label_files):
        x_full_path = os.path.join(train_path, x_path)
        y_full_path = os.path.join(train_path, y_path)

        if not os.path.exists(x_full_path) or not os.path.exists(y_full_path):
            raise ValueError(f'❌ Missing file: {x_path if not os.path.exists(x_full_path) else y_path}')

        x = load_data(x_full_path)
        if x.shape[2] != len(channels):
            raise ValueError(f'❌ Channel mismatch: Expected {len(channels)} channels, found {x.shape[2]} in {x_path}')

        if x.shape[1] != int(fs * win):
            raise ValueError(f'❌ Window size mismatch: Expected {int(fs * win)} samples, found {x.shape[1]} in {x_path}')

def validate_data_structure(parent_path, data_dir, model_channels, model_fs, model_win):
    """
    Validate the data directory structure and ensure it matches the model configuration.
    """
    errors = []
    data_path = os.path.join(parent_path, data_dir)

    # Check if data directory exists
    if not os.path.exists(data_path):
        errors.append(f"Data directory '{data_dir}' not found in '{parent_path}'.")
        return {'errors': errors}

    # Check H5 files
    from seizyml.helper.io import load_data
    h5_files = [f for f in os.listdir(data_path) if f.endswith('.h5')]

    if not h5_files:
        errors.append("No H5 files found in the data directory.")
    else:
        for h5_file in h5_files:
            try:
                data = load_data(os.path.join(data_path, h5_file))
                if data.shape[2] != len(model_channels):
                    errors.append(f"❌ {h5_file}: Channel mismatch (Expected {len(model_channels)}, Found {data.shape[2]}).")
                if data.shape[1] != model_fs * model_win:
                    errors.append(f"❌ {h5_file}: Window size mismatch (Expected {model_fs * model_win}, Found {data.shape[1]}).")
            except:
                errors.append(f"❌ {h5_file}: Failed to load H5 file.")

    return {'errors': errors}

def check_processing_status(parent_path, data_dir, processed_dir, model_predictions_dir):
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
    dict:
        processed_check : bool
            True if the processed directory exists and has the same files as the data directory, False otherwise.
        model_predictions_check : bool
            True if the model predictions directory exists and has the same files as the processed directory, False otherwise.

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
        
    return {'is_processed':processed_check, 'is_predicted':model_predictions_check}

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

