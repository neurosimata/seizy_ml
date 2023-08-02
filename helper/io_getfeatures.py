# -*- coding: utf-8 -*-

### ------------------- Imports ------------------- ###
import os
import tables
import numpy as np
### ----------------------------------------------- ###


def get_data(load_path, load_y=True):
    """
    Retrieve h5 data.

    Parameters
    ----------
    main_path : Str

    Returns
    -------
    data : ndarray (1d = segments, 2d = time, 3d = channels)

    """
    
    # load lfp/eeg data
    f = tables.open_file(load_path, mode = 'r') # open tables object
    data = f.root.data[:]; f.close() # load data
    
    return data


def get_data_with_labels(main_path, exp_path, ch_num=None, inner_path=[], load_y=True):
    """
    Retrieve h5 data with their associated ground truth labels(optional).
    
    --- Examples---
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path=[], load_y = True)
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path={'data_path':'filt_data'}, load_y = False)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    ch_num : List, channels to be loaded, optional, The default is [0,1].
    inner_path : Define inner folder for x or/and y data loading, optional The default is empty list.
    load_y : Bool, If False only data are returned, optional, The default is True.

    Returns
    -------
    data : ndarray (1d = segments, 2d = time, 3d = channels)
    y_data : 1D Int ndarray

    """
    
    if len(inner_path) == 0:
        inner_path = {'data_path':'downsampled_data', 'pred_path':'verified_predictions_pantelis'}
    
    # load lfp/eeg data
    f = tables.open_file(os.path.join(main_path, inner_path['data_path'], exp_path+'.h5'), mode = 'r') # open tables object
    data = f.root.data[:]; f.close() # load data
    
    if ch_num is not None:
        data = data[:,:,ch_num] # get only desired channels
    
    y_data = []
    if load_y == True: # get ground truth labels
        y_data = np.loadtxt(os.path.join(main_path, inner_path['pred_path'], exp_path+'.csv'), delimiter=',', skiprows=0)
        y_data = y_data.astype(np.bool) # convert to bool
        return data, y_data
    
    return data


def save_data(main_path, exp_path, data):
    """
    save_data(main_path, exp_path, data)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    data : ndarray

    Returns
    -------

    """
    
    try:
        # Saving Parameters
        atom = tables.Float64Atom() # declare data type 
        fsave = tables.open_file(os.path.join(main_path, exp_path+'.h5') , mode = 'w') # open tables object
        ds = fsave.create_earray(fsave.root, 'data', atom, # create data store 
                                    [0, data.shape[1], data.shape[2]])
        ds.append(data) # append data
        fsave.close() # close tables object
        return 1
    
    except:
        print('File could not be saved')
        return 0
    
def get_features_single_ch(data, param_list):
    """
    get one channel features metrics 

    Parameters
    ----------
    data : 2D ndarray (1d = segments, 2d = time)
    param_list : ndarray with functions that extract single channgel parameters

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : list, feature names
    """
    
    # create data array
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list.
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list   
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:]) # extract feature
            
    return x_data, feature_labels

def get_features_crossch(data, param_list):
    """
    get cross-channel feature metrics
    
    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    param_list : ndarray with functions that extract cross channel parameters

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : list, feature names
    """
    
    # create data array
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list      
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:,0], data[i,:,1]) # extract feature
        
    return x_data, feature_labels

def get_features_allch(data, param_list, cross_ch_param_list):
    """
    Get features for all channels.

    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    param_list : ndarray with functions that extract single channgel parameters
    cross_ch_param_list : ndarray with functions that extract cross channel parameters

    Returns
    -------
    x_data :  2D ndarray (rows = segments, columns = features)
    labels : np.array, feature names
    """
    
    labels = [] # make list to store labels
    x_data = np.zeros((data.shape[0],0),dtype=np.float) # make array to store all features
    
    # calculate single channel measures
    for ii in range(data.shape[2]):
        # get data and features labels per channel
        temp_data, feature_labels = get_features_single_ch(data[:,:,ii], np.array(param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        
        str2append = '_' + str(ii) # get channel string
        labels += [s + str2append for s in feature_labels] # add to feature labels
    
    # calculcate cross-channel measures
    if len(cross_ch_param_list)>0:
        temp_data, feature_labels = get_features_crossch(data, np.array(cross_ch_param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        labels += [s for s in feature_labels] # add to feature labels   
     
    return x_data, np.array(labels)


