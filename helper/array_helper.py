# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import numpy as np
### ------------------------------------- ###

def contiguous_ones(arr, dur):
    count = 0  # Counts the number of contiguous True sequences

    for val in arr:
        if val:
            count += 1
        else:
            count = 0
        if count > dur: 
            return True
    return False
            

def find_szr_idx(pred_array, dur):
    """
    Identify seizure events and return their start and stop indices, only if they are larger than dur.
    
    Parameters
    ----------
    pred_array : numpy.ndarray, 1D boolean array where `True` indicates a seizure event.
    dur : int, Minimum duration in indices to consider an event valid; effectively acts as erosion.
        
    Returns
    -------
    idx_bounds : numpy.ndarray, 2D array of shape (n_events, 2) containing start and stop indices of valid events.
    
    Examples
    --------
    >>> pred_array = np.array([False, True, False, True, True, False, True])
    >>> find_szr_idx(pred_array, dur=1)
    array([[3, 4]])
    """
    
    # append zeros, find seizure bounds
    ref_pred = np.concatenate((np.zeros(1), pred_array, np.zeros(1)))
    transitions = np.diff(ref_pred)
    rising_edges = np.where(transitions == 1)[0]
    falling_edges = np.where(transitions == -1)[0] - 1 # remove one
    idx_bounds = np.column_stack((rising_edges, falling_edges))
     
    # remove seizures smaller than dur
    idx_length = idx_bounds[:,1] - idx_bounds[:,0]
    idx_bounds = idx_bounds[idx_length >= dur,:]

    return idx_bounds


def merge_close(bounds, merge_margin=5):
    """
    Merge closely spaced seizure segments within a given margin.
    
    Parameters
    ----------
    bounds : numpy.ndarray, 2D array where each row represents start and stop indices of a seizure segment.
    merge_margin : int, Maximum gap between segments to trigger a merge.
        
    Returns
    -------
    numpy.ndarray
        2D array containing start and stop indices of merged segments.
    
    Example
    -------
    >>> merge_close(np.array([[5, 10], [15, 20], [22, 30]]), 3)
    np.array([[5, 10], [15, 30]])
    """

    if bounds.shape[0] < 2:
            return bounds
    
    merged_bounds = []
    current_start, current_end = bounds[0]

    for start, end in bounds[1:]:
        if start - current_end < merge_margin:
            current_end = end  # Extend the current segment
        else:
            merged_bounds.append([current_start, current_end])
            current_start, current_end = start, end  # Start a new segment

    merged_bounds.append([current_start, current_end])  # Append the last segment

    return np.array(merged_bounds)


def match_szrs_idx(bounds_true, y_pred, dur):
    """
    Check for matching seizures in predictions based on ground-truth events.

    Parameters
    ----------
    bounds_true : numpy.ndarray, 2D array of start and stop indices for each true seizure event.
    y_pred : numpy.ndarray, 1D binary array with model's seizure predictions.
    dur : in,  Minimum length of contiguous ones for a match.

    Returns
    -------
    idx : numpy.ndarray
        1D binary array indicating matching seizure events.

    Example
    -------
    >>> match_szrs_idx(np.array([[100, 150], [200, 250]]), np.array([0, 1, ...]), 2)
    np.array([1, 0])
    """
    idx = np.zeros(bounds_true.shape[0])
     
    for i in range(bounds_true.shape[0]):
        pred = y_pred[bounds_true[i, 0]:bounds_true[i, 1] + 1]
        idx[i] = contiguous_ones(pred, dur)

    return idx.astype(bool)























            
            


