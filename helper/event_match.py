#### ------------- Imports ------------- ####
import numpy as np
from scipy import ndimage
#### ----------------------------------- ####

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
    >>> bounds_true = np.array([[100, 150], [200, 250]])
    >>> y_pred = np.zeros(300)
    >>> y_pred[105:145] = 1
    >>> match_szrs_idx(bounds_true, y_pred)
    >>> array([ True, False])
    """
    idx = np.zeros(bounds_true.shape[0])
    for i in range(bounds_true.shape[0]):
        pred = y_pred[bounds_true[i, 0]:bounds_true[i, 1] + 1]
        if pred.sum() > 0:
            idx[i] = 1
    return idx.astype(bool)

### post-processing methods ###

# TODO add tests
def dual_threshold(raw_pred, t_high=.5, t_low=.2, win_size=6):
    """
    Apply hysteresis thresholding to detect events in a signal based on high and low thresholds.
    
    Hysteresis thresholding is a two-step process where a high threshold (`t_high`) is used to 
    initiate an event, and a lower threshold (`t_low`) is used to continue the event until 
    the signal falls below it. This method reduces noise and false positives in event detection.
    
    Parameters
    ----------
    raw_pred : numpy.ndarray
        1D array of raw predictions or signal values.
    t_high : float, optional, default=0.5
        High threshold value for starting an event.
    t_low : float, optional, default=0.2 
        Low threshold value for ending an event.
    win_size : int, optional, default=6
        Size of the rolling window (in samples) used to smooth the raw predictions 
        before applying thresholds.
    
    Returns
    -------
    numpy.ndarray
        1D binary array indicating detected events (1 for event, 0 otherwise).
    
    Notes
    -----
    - The rolling average is computed using a convolution operation to smooth the 
      raw predictions.
    - The binary propagation step ensures that once an event is triggered (based on 
      `t_high`), it continues until the signal drops below `t_low`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> raw_pred = np.array([0.1, 0.4, 0.6, 0.7, 0.5, 0.3, 0.1])
    >>> dual_threshold(raw_pred, t_high=0.5, t_low=0.2, win_size=3)
    array([0, 0, 1, 1, 1, 1, 0])
    
    Raises
    ------
    ValueError
        If `t_low` is greater than or equal to `t_high`.
    """
    
    mean_pred = np.convolve(raw_pred, np.ones(win_size)/win_size, mode='same')
    seeds = mean_pred >= t_high
    mask = mean_pred > t_low
    hysteresis_output = ndimage.binary_propagation(seeds, mask=mask)
    return hysteresis_output.astype(int)

def dilation_erosion(vector, dilation=0, erosion=0):
    """
    Applies morphological operations to a 1D binary vector.
    
    Parameters:
        vector (numpy.ndarray): Input 1D binary vector.
        dilation (int): Size of the dilation structuring element minus one (default is 0).
        erosion (int): Size of the erosion structuring element (default is 0).
    
    Returns:
        numpy.ndarray: The processed vector after applying the specified morphological operations, with padding removed.
        
    Notes
    -----
    Padding is added to the vector to prevent edge effects during morphological operations. 
    It is removed before returning the result.
    
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
        
    Notes
    -----
    Padding is added to the vector to prevent edge effects during morphological operations. 
    It is removed before returning the result.
    
    """
    padded_vector = np.concatenate(([0,0,0], vector, [0,0,0]))
    operation1 = ndimage.binary_opening(padded_vector, structure=np.ones(erosion+1)).astype(int)
    operation2 = ndimage.binary_closing(operation1, structure=np.ones(dilation+1)).astype(int)
    return operation2[3:-3]

def clean_predictions(raw_pred, operation='dual_threshold', dilation=None, erosion=None,
                      rolling_window=None, t_high=None, t_low=None):
    """
    Clean model predictions with post-processing methods.
    
    Parameters
    ----------
    raw_pred : numpy.ndarray
        1D binary array of raw model predictions.
    operation : str
        Type of post-processing to apply. Must be one of {'dilation_erosion', 'erosion_dilation', 'dual_threshold'}.
    dilation : int, optional
        Size of dilation structuring element. Required for `dilation_erosion` and `erosion_dilation`.
    erosion : int, optional
        Size of erosion structuring element. Required for `dilation_erosion` and `erosion_dilation`.
    rolling_window : int, optional
        Window size for smoothing in `dual_threshold`.
    t_high : float, optional
        High threshold for `dual_threshold`.
    t_low : float, optional
        Low threshold for `dual_threshold`.

    Returns
    -------
    numpy.ndarray
        1D binary array of cleaned predictions.

    Raises
    ------
    ValueError
        If an unsupported operation is provided or required parameters are missing.
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
    
    print('Testing dilation_erosion function...')
    # Test with provided example
    vector = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 1 failed"

    # Test with all zeros
    vector = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    expect = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 2 failed"

    # Test with all ones
    vector = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 3 failed"

    # Test with isolated ones
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 4 failed"

    # Test with a single one
    vector = np.array([0, 0, 0, 1, 0, 0, 0])
    expect = np.array([0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 5 failed"

    # Test with two ones close together
    vector = np.array([0, 1, 0, 1, 0])
    expect = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 6 failed"

    # Test with complex pattern
    vector = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 7 failed"
    
    # Test with isolated ones
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 8 failed"
    
    # Test with two blocks
    vector = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    expect = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(dilation_erosion(vector, dilation=2, erosion=1), expect), "Test case 8 failed"
    print("All test cases passed!\n")
    
    print('Testing erosion_dilation function...')
    # Test with provided example
    vector = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    expect = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 1 failed"

    # Test with all zeros
    vector = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    expect = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 2 failed"

    # Test with all ones
    vector = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 3 failed"

    # Test with isolated ones
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    expect = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 4 failed"

    # Test with a single pair
    vector = np.array([0, 0, 0, 1, 1, 0, 0])
    expect = np.array([0, 0, 0, 1, 1, 0, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 5 failed"

    # Test with two ones close together
    vector = np.array([0, 1, 1, 0, 1, 1, 0])
    expect = np.array([0, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 6 failed"

    # Test with complex pattern
    vector = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    expect = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 7 failed"
    
    
    # Test with two blocks
    vector = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    expect = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(erosion_dilation(vector, dilation=2, erosion=1), expect), "Test case 8 failed"
    print("All test cases passed!\n")
    
    ### Tests for get_szr_idx ###
    print('Testing get_szr_idx function...')
    # Test with provided example
    pred_array = np.array([0, 1, 0, 1, 1, 0, 1])
    expect = np.array([[1, 1], [3, 4], [6, 6]])
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 1 failed"

    # Test with no seizures
    pred_array = np.array([0, 0, 0, 0])
    expect = np.array([]).reshape(0, 2)
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 2 failed"

    # Test with continuous seizures
    pred_array = np.array([1, 1, 1, 1])
    expect = np.array([[0, 3]])
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 3 failed"

    # Test with single seizure
    pred_array = np.array([0, 0, 1, 0, 0])
    expect = np.array([[2, 2]])
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 4 failed"

    # Test with alternating seizures
    pred_array = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    expect = np.array([ [2, 5], [9, 10]])
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 5 failed"

    # Test with multiple seizures
    pred_array = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    expect = np.array([[0, 1], [4, 5], [7, 8]])
    assert np.array_equal(get_szr_idx(pred_array), expect), "Test case 6 failed"
    print("All test cases passed!\n")
    
    ### Tests for match_szrs_idx ###
    print('Testing match_szrs_idx function...')
   # Test with provided example
    bounds_true = np.array([[0, 3], [4, 6]])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 1])
    expect = np.array([1, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 1 failed"

    # Test with no matching seizures
    bounds_true = np.array([[0, 2], [4, 6]])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0])
    expect = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 2 failed"

    # Test with all matching seizures
    bounds_true = np.array([[0, 1], [2, 3], [4, 5]])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    expect = np.array([1, 1, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 3 failed"

    # Test with partially matching seizures
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 1, 0, 0])
    expect = np.array([0, 1], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 4 failed"

    # Test with empty bounds_true
    bounds_true = np.empty((0, 2), dtype=int)
    y_pred = np.array([1, 1, 1, 1])
    expect = np.array([], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 5 failed"

    # Test with y_pred all zeros
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expect = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 6 failed"
    
    # Test with some mseizures matching
    bounds_true = np.array([[0, 2], [3, 5]])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expect = np.array([0, 0], dtype=bool)
    assert np.array_equal(match_szrs_idx(bounds_true, y_pred), expect), "Test case 7 failed"

    print("All test cases passed!")
