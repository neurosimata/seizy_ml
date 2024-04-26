# -*- coding: utf-8 -*-

### ------------------- Imports ------------------- ###
from helper import features
import numpy as np
### ----------------------------------------------- ###
    
def compute_features(data, single_channel_functions, channel_names):
    """
    Compute features from given 3D data.
    
    Parameters
    ----------
    data : ndarray, 3D array of shape (segments, time, channels) containing the input data.
    single_channel_functions : List of function names for single channel features, defined in features.py.
    channel_names : List of channel names used to name the features.

    Returns
    -------
    features_array : 2D array containing the computed features for each segment, with shape (segments, num_features).
    feature_labels : Array of labels for the computed features, corresponding to the columns in features_array.
    """
    num_segments, _, num_channels = data.shape
    feature_labels = []
    features_list = []

    # Calculating single channel features
    for i, func_name in enumerate(single_channel_functions):
        func = getattr(features, func_name)
        for c in range(num_channels):
            feature_name = f"{func_name}-{channel_names[c]}"
            feature_labels.append(feature_name)
            features_list.append([func(data[s, :, c]) for s in range(num_segments)])

    features_array = np.column_stack(features_list)  # Combining the feature arrays into a single 2D array
    return features_array, np.array(feature_labels)

def compute_selected_features(data, selected_feature_names, channel_names):
    """
    Compute selected features from given 3D data based on feature names.

    Parameters
    ----------
    data : 3D array of shape (segments, time, channels) containing the input data.
    selected_feature_names : List of selected feature names, must match the naming convention used in compute_features.
    channel_names : List of channel names used to match the features.

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


