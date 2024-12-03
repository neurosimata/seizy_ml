import numpy as np
from scipy import ndimage

def clean_predictions_de(vector):
    """
    Performs dilation followed by erosion on a 1D binary vector.
    
    This function first applies binary closing to merge `1`s that are separated
    by up to two `0`s. Then, it performs binary opening to remove isolated `1`s
    surrounded by `0`s.
    
    Parameters:
    vector (numpy.ndarray): Input 1D binary vector.
    
    Returns:
    numpy.ndarray: The processed vector after dilation and erosion.
    """
    padded_vector = np.concatenate(([0], vector, [0]))
    closed = ndimage.binary_closing(padded_vector, structure=[1, 1, 1]).astype(int)
    opened = ndimage.binary_opening(closed, structure=[1, 1]).astype(int)
    return opened[1:-1]

def get_szr_idx(pred_array):
    """
    Identify seizure events and return their start and stop indices.
    
    Parameters
    ----------
    pred_array : numpy.ndarray, 1D boolean array where `True` indicates a seizure event.
        
    Returns
    -------
    idx_bounds : numpy.ndarray, 2D array of shape (n_events, 2) containing start and stop indices of valid events.
    
    Examples
    --------
    >>> pred_array = np.array([False, False, False, True, True, False, False])
    >>> find_szr_idx(pred_array)
    array([[3, 4]])
    """
    
    ref_pred = np.concatenate(([0], pred_array, [0]))
    transitions = np.diff(ref_pred)
    rising_edges = np.where(transitions == 1)[0]
    falling_edges = np.where(transitions == -1)[0] - 1
    idx_bounds = np.column_stack((rising_edges, falling_edges))

    return idx_bounds

def match_szrs_idx(bounds_true, y_pred):
    """
    Check for matching seizures in predictions based on ground-truth events.

    Parameters
    ----------
    bounds_true : numpy.ndarray, 2D array of start and stop indices for each true seizure event.
    y_pred : numpy.ndarray, 1D binary array with model's seizure predictions.

    Returns
    -------
    idx : numpy.ndarray
        1D binary array indicating matching seizure events.

    Example
    -------
    >>> match_szrs_idx(np.array([[100, 150], [200, 250]]), np.array([0, 1, ...]))
    np.array([1, 0])
    """
    idx = np.zeros(bounds_true.shape[0])
     
    for i in range(bounds_true.shape[0]):
        pred = y_pred[bounds_true[i, 0]:bounds_true[i, 1] + 1]
        if pred.sum() > 0:
            idx[i] = 1
    return idx.astype(bool)

def dual_threshold(raw_pred, t_high=.5, t_low=.2, win_size=6):
    """
    Apply hysteresis thresholding to detect events in a signal.

    Parameters:
    raw_pred (np.array): Raw model predictions (binary)
    t_high (float): The high threshold for triggering an event
    t_low (float): The low threshold for ending an event

    Returns:
    np.array: Binary event vector after hysteresis thresholding
    """
    
    mean_pred = np.convolve(raw_pred, np.ones(win_size)/win_size, mode='same')
    seeds = mean_pred >= t_high
    mask = mean_pred > t_low
    hysteresis_output = ndimage.binary_propagation(seeds, mask=mask)
    return hysteresis_output.astype(int)

# post-processing methods
def dilation_erosion(vector, dilation=0, erosion=0):
    """
    Applies morphological operations to a 1D binary vector.
    
    Parameters:
        vector (numpy.ndarray): Input 1D binary vector.
        dilation (int): Size of the dilation structuring element minus one (default is 0).
        erosion (int): Size of the erosion structuring element (default is 0).
    
    Returns:
        numpy.ndarray: The processed vector after applying the specified morphological operations, with padding removed.
    
    """
    padded_vector = np.concatenate(([0,0,0], vector, [0,0,0]))
    operation1 = ndimage.binary_closing(padded_vector, structure=np.ones(dilation+1)).astype(int)
    operation2 = ndimage.binary_opening(operation1, structure=np.ones(erosion+1)).astype(int)
    return operation2[3:-3]

def erosion_dilation(vector, dilation=0, erosion=0):
    """
    Applies morphological operations to a 1D binary vector.
    
    Parameters:
        vector (numpy.ndarray): Input 1D binary vector.
        dilation (int): Size of the dilation structuring element minus one (default is 0).
        erosion (int): Size of the erosion structuring element (default is 0).
    
    Returns:
        numpy.ndarray: The processed vector after applying the specified morphological operations, with padding removed.
    
    """
    padded_vector = np.concatenate(([0,0,0], vector, [0,0,0]))
    operation1 = ndimage.binary_opening(padded_vector, structure=np.ones(erosion+1)).astype(int)
    operation2 = ndimage.binary_closing(operation1, structure=np.ones(dilation+1)).astype(int)
    return operation2[3:-3]

def clean_predictions(raw_pred, operation='dual_threshold', dilation=None, erosion=None,
                      rolling_window=None, t_high=None, t_low=None):
    """
    Higher level method to clean predictions called by CLI

    Parameters
    ----------
    raw_pred : np.array, binary model predictions
    operation : str, type of post-processing operation to 

    Returns
    -------
    clean_pred :  np.array: Clean model predictions
    """
    if operation == 'dilation_erosion':
        clean_pred = dilation_erosion(raw_pred, dilation=dilation, erosion=erosion)
    elif operation == 'erosion_dilation':
        clean_pred = erosion_dilation(raw_pred, erosion=erosion, dilation=dilation)
    elif operation == 'dual_threshold':
        clean_pred = dual_threshold(raw_pred, win_size=rolling_window, t_high=t_high, t_low=t_low,)
    else:
        raise(f'Post-proccessing method {operation} was not found.')
    return clean_pred


if __name__ == '__main__':
    
    ### tests for clean predictions ###
    
    print('Testing clean predictions function...')
    # Test with provided example
    vector = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 1 failed"

    # Test with all zeros
    vector = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 2 failed"

    # Test with all ones
    vector = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 3 failed"

    # Test with isolated ones
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 4 failed"

    # Test with a single one
    vector = np.array([0, 0, 0, 1, 0, 0, 0])
    expected = np.array([0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 5 failed"

    # Test with two ones close together
    vector = np.array([0, 1, 0, 1, 0])
    expected = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 6 failed"

    # Test with complex pattern
    vector = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 7 failed"
    
    # Test with isolated ones
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0])
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 8 failed"
    
    # Test with two blocks
    vector = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    expected = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(clean_predictions_de(vector), expected), "Test case 8 failed"
    print("All test cases passed!\n")
    
    ### Tests for get_szr_idx ###
    print('Testing get_szr_idx function...')
    # Test with provided example
    pred_array = np.array([0, 1, 0, 1, 1, 0, 1])
    expected = np.array([[1, 1], [3, 4], [6, 6]])
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 1 failed"

    # Test with no seizures
    pred_array = np.array([0, 0, 0, 0])
    expected = np.array([]).reshape(0, 2)
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 2 failed"

    # Test with continuous seizures
    pred_array = np.array([1, 1, 1, 1])
    expected = np.array([[0, 3]])
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 3 failed"

    # Test with single seizure
    pred_array = np.array([0, 0, 1, 0, 0])
    expected = np.array([[2, 2]])
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 4 failed"

    # Test with alternating seizures
    pred_array = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    expected = np.array([ [2, 5], [9, 10]])
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 5 failed"

    # Test with multiple seizures
    pred_array = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    expected = np.array([[0, 1], [4, 5], [7, 8]])
    assert np.array_equal(get_szr_idx(pred_array), expected), "Test case 6 failed"
    print("All test cases passed!\n")
    
    ### Tests for match_szrs_idx ###
    print('Testing match_szrs_idx function...')
   # Test with provided example
    bounds_true = np.array([[0, 3], [4, 6]])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 1])
    expected = np.array([1, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 1 failed"

    # Test with no matching seizures
    bounds_true = np.array([[0, 2], [4, 6]])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0])
    expected = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 2 failed"

    # Test with all matching seizures
    bounds_true = np.array([[0, 1], [2, 3], [4, 5]])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    expected = np.array([1, 1, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 3 failed"

    # Test with partially matching seizures
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 1, 0, 0])
    expected = np.array([0, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 4 failed"

    # Test with empty bounds_true
    bounds_true = np.empty((0, 2), dtype=int)
    y_pred = np.array([1, 1, 1, 1])
    expected = np.array([], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 5 failed"

    # Test with y_pred all zeros
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expected = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 6 failed"
    
    # Test with some mseizures matching
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expected = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expected), "Test case 7 failed"

    print("All test cases passed!")
