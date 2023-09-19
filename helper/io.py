# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import tables
### ------------------------------------- ###

def load_data(load_path):
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

def save_data(save_path, data):
    """
    Save h5 data

    Parameters
    ----------
    save_path : Str
    data : ndarray

    Returns
    -------

    """
    
    try:
        # Saving Parameters
        atom = tables.Float64Atom() # declare data type 
        fsave = tables.open_file(save_path , mode='w') # open tables object
        
        # create data store 
        if len(data.shape) == 3:
            ds = fsave.create_earray(fsave.root, 'data', atom,
                                        [0, data.shape[1], data.shape[2]])
        elif len(data.shape) == 2:
            ds = fsave.create_earray(fsave.root, 'data', atom,
                                        [0, data.shape[1]])
        ds.append(data) # append data
        fsave.close() # close tables object
        return 1
    
    except Exception as e:
        print('File could not be saved')
        print(e)
        return 0