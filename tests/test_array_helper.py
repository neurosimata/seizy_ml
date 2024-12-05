####-------------------------------- Imports --------------------------- ######
import pytest
import numpy as np
from helper.event_match import get_szr_idx, match_szrs_idx
####------------------------------- Fixtures --------------------------- ######

def create_bin_array(length, index):
    """
    Create binary array based on index.

    Parameters
    ----------
    length : int, array length
    index : 2D list, containing indx[[start1,stop1], [start2,stop2]]

    Returns
    -------
    arr : array, binary
    
    """
    arr = np.zeros(length)
    for i in index:
        arr[i[0]: i[1]+1] = 1
    return arr

@pytest.fixture
def properties():
    prop = {}
    return prop

@pytest.fixture
def create_binary_array():
    return create_bin_array


####--------------------------------------------------- ######


####---------------------------- Tests -------------------------- ######
@pytest.mark.parametrize("length, index", 
                          [(100, [[1, 5], [11, 15]]), 
                           (100, [[0, 4], [30, 60]]),
                           (100, [[1, 2], [15, 20], [90, 99]]),
                          ])
def test_get_szr_index(create_binary_array, length, index):
    
    # create array and find original bounds
    arr = create_binary_array(length, index)
    szr_bounds = get_szr_idx(arr)
    assert np.allclose(szr_bounds, np.array(index))

@pytest.mark.parametrize("length, index_true, index_pred, matching", 
                          [(100, [[1, 5], [11, 15], [50, 51]], [[4, 5], [12, 14]], 2), 
                           (100, [[0, 4], [30, 60]],  [[3, 4], [12, 15]], 1),
                           (100, [[0, 4], [30, 60]],  [[6, 7], [12, 15]], 0),
                           (100, [[1, 10], [15, 20], [90, 94]], [[1, 5], [18, 18], [91, 94]], 3),
                          ])
def test_get_szr_match_szrs_idx(create_binary_array, length,
                                 index_true, index_pred, matching):
    
    # get true bounds and pred array
    true_arr = create_binary_array(length, index_true)
    true_bounds = get_szr_idx(true_arr)
    pred_arr = create_binary_array(length, index_pred)
    
    # find matching index
    idx = match_szrs_idx(true_bounds, pred_arr)
    assert idx.sum() == matching


@pytest.mark.parametrize("length, index", 
                          [ (100, [[0, 99]]),  # Edge Case: Single segment
                            (100, [[0, 0], [99, 99]])  # Edge Case: 1's at edges
                          ])
def test_get_szr_index_edge_cases(create_binary_array, length, index):
    arr = create_binary_array(length, index)
    szr_bounds = get_szr_idx(arr)
    assert np.allclose(szr_bounds, np.array(index))





