# -*- coding: utf-8 -*-

### -------------------------------- IMPORTS ------------------------------ ###
import os
import numpy as np
from tqdm import tqdm
from helper.io_getfeatures import get_data, save_data
from scipy.signal import butter, zpk2sos, sosfiltfilt
### ------------------------------------------------------------------------###


###### --------------------- Batch process functions ----------------- ######

class PreProcess:
    """
    Remove outliers and filer data from parent folder.
    """
    
    def __init__(self, load_path, save_path, fs,
                 outier_threshold=25, freq_cutoff=[2]):
        """
        

        Parameters
        ----------
        load_path : str
        save_path : str
        fs : int
        outier_threshold : int/float
        freq_cutoff : list, containing a numeral for hpfilter cutoff.

        Returns
        -------
        None.

        """

        self.load_path = load_path
        self.save_path = save_path
        self.fs = fs
        self.outier_threshold = outier_threshold
        self.freq_cutoff = freq_cutoff

    def filter_data(self):
        """
        Preprocess data in parent folder.
    
        """
        
        # create save path if it doesn't exist
        if os.path.isdir(self.save_path) == 0:
            os.mkdir(self.save_path)
        
        # get file list 
        filelist = list(filter(lambda k: '.h5' in k, os.listdir(self.load_path)))
        
        print('\n --->', len(filelist), 'files will be processed.\n')
        for i in tqdm(range(0, len(filelist)), desc = 'Progress:'): # loop through experiments 
        
            # clean and filter data
            data = get_data(os.path.join(self.load_path, filelist[i]))
            data = self.filter_clean(data)
            
            # save clean data
            save_data(self.save_path, filelist[i], data)
            
        print('Files in', self.load_path, 'directory have been cleaned and saved in:',
              '-', self.save_path, '-')
        print('---------------------------------------------------------------------------\n')
            
    
    def filter_clean(self, data):
        """
        Filter and remove outliers.
         
        data : 3d ndarray, (1d = segments, 2d = time, 3d = channels)
        clean : bool, if true remove outliers
        filt : bool, if true filter data
    
        Returns
        -------
        data : 3d ndarray, (1d = segments, 2d = time, 3d = channels)
        """
        dim = data.shape # get data dimensions
        
    
        if self.outier_threshold is not None: # if true remove outliers
            for i in range(data.shape[2]):
                temp = clean_signal(data[:,:,i].flatten(), threshold=self.outier_threshold)
                data[:,:,i] = temp.reshape((dim[0], dim[1]))
    
        if self.freq_cutoff is not None:
            for i in range(data.shape[2]):
                # high-pass filter data
                data[:,:,i] = batch_filter(data[:,:,i], butter_highpass, fs=self.fs, freq_cutoff=self.freq_cutoff)
        return data


def clean_signal(data, threshold=25):
    """
    Removes outliers and replaces with median

    Parameters
    ----------
    data : 1D signal, numpy array
    threshold : Real number, The default is 25.

    Returns
    -------
    clean_data : 1D signal, numpy array

    """
    
    clean_data = np.copy(data)
    clean_data = clean_data - np.mean(clean_data)
    idx = np.where(np.abs(data) > (threshold * np.std(data)))
    clean_data[idx] = np.median(clean_data)
    
    return clean_data

def batch_filter(data, filt_func, freq_cutoff, fs=100, verbose=False):
    """
    batch_filter(data)

    Parameters
    ----------
    data : 2d numpy array
    filt_func: filter object
    freq_cutoff: list with frequency cutoff(s)
    verbose: bool, if True verbose

    Returns
    -------
    filt_data : 2d numpy array

    """
    
    # Init data matrix
    filt_data = np.zeros(data.shape)
    filt_data = filt_data.astype(np.double)
    
    if verbose == True:
        for i in tqdm(range(data.shape[0])):
            filt_data[i,:] = filt_func(data[i,:], freq_cutoff, fs=fs)
            
    elif verbose == False:
        for i in range(data.shape[0]):
            filt_data[i,:] = filt_func(data[i,:], freq_cutoff, fs=fs)
        
        
    return filt_data

def butter_highpass(data, cutoff, fs, order=5):
    """
    butter_highpass(data, cutoff, fs, order = 5)

    Parameters
    ----------
    data : 1d ndarray, signal 
    cutoff : Float, cutoff frequency
    fs : Int, sampling rate
    order : Int, filter order. The default is 5.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    nyq = 0.5 * fs               # Nyquist Frequency (Hz)
    normal_cutoff = cutoff[0] / nyq # Low-bound Frequency (Normalised)
    
    # Design filter
    z,p,k = butter(order, normal_cutoff, btype='high', analog=False, output ='zpk') 
    
    sos = zpk2sos(z,p,k)         # Convert to second order sections
    y = sosfiltfilt(sos, data)   # Filter data
    return y





