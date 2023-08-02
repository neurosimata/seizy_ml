# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import numpy as np
from numba import jit
from scipy import signal
### ------------------------------------- ###


def chunk_array(start, stop, div):
    """
    Get array to read labchart in chunks.
    
    Parameters
    ----------
    start : int
    stop : int
    div : int,

    Returns
    -------
    idx : numpy array, 1st col = start, 2nd col = stop

    """

    rem = (stop-start+1) % div
    trim_stop = (stop - rem)
    temp_idx = np.linspace(start, trim_stop, round(trim_stop/div)+1, dtype=int)
    temp_idx = np.append(temp_idx, stop)
    temp_idx = np.unique(temp_idx)
    
    # reshape into 2 column format
    idx = np.zeros((len(temp_idx)-1, 2), dtype=int)
    idx[:, 0] = temp_idx[:-1]
    idx[:, 1] = temp_idx[1:]
    idx[1:, 0] += 1
    return idx

@jit(nopython=True)
def find_value(array, value, start=1, order=1):
    """
    Find exact value in array.
    
    --- Examples ---
    a = np.array([1.5, 2,.5, 3.5, 5, 15])
    idx = find_value(a, 2, start = 1, order = 1)
    
    Parameters
    ----------
    array : array
    value : int/float, value to be found.
    start = int, starting index
    order : 1, search forwards
          :-1, search reverse

    Returns
    -------
    index, array
    """ 

    if order == 1:
        for i in range(start, len(array)):
            if array[i] == value:
                return int(i)
            
    elif order == -1:
        for i in range(start, -1, -1):
            if array[i] == value:
                return int(i)
            
@jit(nopython=True)               
def remove_zeros(ref_pred, pred_array, bounds):
    """   
    Replace values with zeros.
    
    Parameters
    ----------
    ref_pred : ndarray, boolean array
    pred_array : ndarray, boolean array
    bounds : list, (elements remove before, after detected seizure)

    Returns
    -------
    ref_pred : ndarray (szr_n,2), neighbor threshold

    """
    # remove neighbours that have zeros
    for i in range(bounds[0], pred_array.shape[0] - bounds[1]):
        if np.sum(pred_array[i-bounds[0]:i+bounds[1]+1]) != np.sum(bounds)+1:
            ref_pred[i] = 0
    return ref_pred
            

def find_szr_idx(pred_array, dur=0):
    """
    find seizure bounds.
    idx_bounds = find_szr_idx(rawpred, np.array([2,2]))
    
    Parameters
    ----------
    pred_array : ndarray, boolean array
    bounds: two element list, denoting neighbours bounds 
       
    Returns
    -------
    idx_bounds : NUMPY ARRAY (szr_n,2)
        index bounds for seizures.
    
    """
    
    # make a copy of the array
    ref_pred = np.copy(pred_array)
    
    # get min peak distance
    distance = 1
    
    # append 1 to beginning and end
    ref_pred = np.concatenate((np.zeros(1), ref_pred, np.zeros(1)))
    
    # get signal peaks        
    idx = signal.find_peaks(ref_pred, height = 1, distance = distance)[0]
  
    # get index bounds
    idx_bounds = np.zeros([len(idx),2], dtype=int)
   
    for i in range(len(idx)):
        idx_bounds[i,0] = find_value(ref_pred, 0, start = idx[i], order = -1) + 1
        idx_bounds[i,1] = find_value(ref_pred, 0, start = idx[i], order = 1) - 1
     
    # remove seizures smaller than dur   
    idx_length = idx_bounds[:,1] - idx_bounds[:,0]
    idx_bounds = idx_bounds[idx_length>=dur,:]
    
    # get original index
    idx_bounds = idx_bounds-1
    
    return idx_bounds


def merge_close(bounds, merge_margin = 5):
    """
    merge_close(bounds, merge_margin = 5)

    Parameters
    ----------
    bounds : 2D ndarray (rows = seizure segments, columns =[start,stop])
    merge_margin : Int, optional

    Returns
    -------
    bounds_out : 2D ndarray, merged array (rows = seizure segments, columns =[start,stop])

    """
    
    if bounds.shape[0] < 2: # if less than two seizures exit
        return bounds
    
    # copy of bounds
    bounds_out = np.copy(bounds) 

    # find bounds separated by less than merge_margin
    delta = bounds[1:,0] - bounds[:-1,1]
    merge_idx = delta < merge_margin; 
    
    # padd with zeros for peak detection
    element = np.zeros(1, dtype=bool)
    merge_idx = np.concatenate((element, merge_idx, element))
    merge_idx = find_szr_idx(merge_idx)
    merge_idx[:, 1] +=1
    merge_idx -= 1  # (-1 for extra addition at 0 element)
    
    # make a copy and leave unchanged, index for original array
    idx = np.copy(merge_idx)

    for i in range(merge_idx.shape[0]):                                             # loop though index
        low = merge_idx[i,0]; upper = merge_idx[i,1]                                # get upper and lower boundaries
        bounds_out[ merge_idx[i,0],:] = [bounds[idx[i,0],0], bounds[idx[i,1],1]]    # replace merged  
        rmv_idx = np.linspace(low, upper, int(upper-low)+1)                         # get removal index
        rmv_idx = np.delete(rmv_idx,0).astype(np.int64)                             # remove first element and convert to int
        bounds_out = np.delete(bounds_out, rmv_idx , axis=0)                        # delete next element
        merge_idx -= rmv_idx.shape[0]                                               # remove one from index because of deleted element 
        
    return bounds_out
        

# find matching seizures     
@jit(nopython=True) 
def match_szrs(idx_true, idx_pred, err_margin = 5):
    """
    match_szrs(idx_true,idx_pred, err_margin)

    Parameters
    ----------
    idx_true : Bool, ndarray, User defined (ground truth) boolean index
    idx_pred : Bool, ndarray, Predicted index
    err_margin : int, optional, Default values = 5.

    Returns
    -------
    matching : int, number of matching seizures

    """
    matching = 0 # number of matching seizures
    
    for i in range(idx_true.shape[0]):
        
        # does min bound match within error margin?
        min_bound = np.any(np.abs(np.subtract(idx_true[i,0],idx_pred[:,0]))<err_margin)
        
        # does max bound match within error margin?
        max_bound  = np.any(np.abs(np.subtract(idx_true[i,1],idx_pred[:,1]))<err_margin)
        
        # do both bounds match?
        if max_bound is True & min_bound is True:
            matching += 1
            
    return matching

# find matching seizures method 2 with index (preferred method)
@jit(nopython=True) 
def match_szrs_idx(bounds_true, y_pred, bounds):
    """
    find index of matching seizures.
    
    Parameters
    ----------
    bounds_true : np.array, index of true seizures  
    y_pred : np.array, binary predictions of model
    bounds : np.array, (elements to remove before or after seizure)
    
    Returns
    -------
    idx, containing ones or zeros
    
    """
    # create empty vector
    idx = np.zeros(bounds_true.shape[0])
    
    for i in range(bounds_true.shape[0]):
        
        # get predictions in seizure range
        pred = y_pred[bounds_true[i,0]:bounds_true[i,1] + 1]
        
        # get sum of continous predictions > more than 10 seconds
        sum_continous_segments = np.sum(remove_zeros(pred.copy(),
                                                     pred, bounds))
        
        # pass to index array
        idx[i] = sum_continous_segments
 
    return idx > 0 # convert to logic






















            
            


