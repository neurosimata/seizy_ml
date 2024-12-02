# -*- coding: utf-8 -*-

### ------------------- Imports ------------------- ###
from helper import features
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
n_jobs = int(multiprocessing.cpu_count() * 0.8)
### ----------------------------------------------- ###

def compute_features(data, single_channel_functions, channel_names, fs):
    """
    Compute features from given 3D data.
    
    Parameters
    ----------
    data : ndarray, 3D array of shape (segments, time, channels) containing the input data.
    single_channel_functions : List of function names for single channel features, defined in features.py.
    channel_names : List of channel names used to name the features.
    fs: int, sampling rate

    Returns
    -------
    features_array : 2D array containing the computed features for each segment, with shape (segments, num_features).
    feature_labels : Array of labels for the computed features, corresponding to the columns in features_array.
    """
    
    # get dims and init lists
    num_segments, _, num_channels = data.shape
    feature_labels = []
    features_list = []

    # Calculate single channel features
    param_list = [getattr(features, func_name) for func_name in single_channel_functions]
    def extract_features_for_segment(segment, param_list):
        return [func(segment, fs=fs) for func in param_list]
        
    for c in range(num_channels):
        x_data = Parallel(n_jobs=n_jobs)(delayed(extract_features_for_segment)(segment, param_list) for segment in data[:,:,c])
        features_list.extend(x_data)
        feature_labels.extend([f"{func_name}-{channel_names[c]}" for func_name in single_channel_functions])

    features_array = np.column_stack(features_list)
    return features_array, np.array(feature_labels)

def compute_selected_features(data, selected_feature_names, channel_names, fs):
    """
    Compute selected features from given 3D data based on feature names.

    Parameters
    ----------
    data : 3D array of shape (segments, time, channels) containing the input data.
    selected_feature_names : List of selected feature names, must match the naming convention used in compute_features.
    channel_names : List of channel names used to match the features.
    fs: int, sampling rate
    
    Returns
    -------
    features_array : 2D array containing the computed features for each segment, with shape (segments, num_selected_features).
    feature_labels : Array of labels for the computed features, corresponding to the columns in features_array.
    """
    num_segments, _, num_channels = data.shape
    feature_labels = []
    features_list = []

    for feature_name in selected_feature_names:
        parts = feature_name.split('-')
        func_name = parts[0]
        channel_indices = [channel_names.index(ch_name) for ch_name in parts[1:]]

        func = getattr(features, func_name)
        if len(channel_indices) == 1:
            c = channel_indices[0]
            features_list.append([func(data[s, :, c]) for s in range(num_segments)])
        elif len(channel_indices) == 2:
            c1, c2 = channel_indices
            features_list.append([func(data[s, :, c1], data[s, :, c2]) for s in range(num_segments)])

        feature_labels.append(feature_name)

    features_array = np.column_stack(features_list)
    return features_array, np.array(feature_labels)


