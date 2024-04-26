# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import numpy as np
from numba import jit
from scipy.fftpack import fft
### ------------------------------------- ###


def psd(signal, fs=256, freq=[2,40]):
    """
    Measure the power of the signal over the signal over 2-40 Hz.
    
    Parameters
    ----------
    signal : 1D numpy array
    fs : int, sampling rate
    freq: list(low, high)
    
    Returns
    -------
    power_area : np.float
    
    """
    
    # get frequency Boundaries
    f_res = signal.shape[0]/fs
    flow = int(freq[0]*f_res)
    fup = int(freq[1]*f_res) + 1
       
    # get power spectrum and normalize to signal length
    xdft = np.square(np.absolute(fft(signal)))
    xdft = xdft * (1/(fs*signal.shape[0]))
       
    # multiply *2 to conserve energy in positive frequencies
    psdx = 2*xdft[0:int(xdft.shape[0]/2+1)] 
    return np.sum(psdx[flow:fup])

def delta_power(signal):
    return psd(signal, fs=256, freq=[1, 4])

def alpha_power(signal):
    return psd(signal, fs=256, freq=[4.2, 8])

def theta_power(signal):
    return psd(signal, fs=256, freq=[8.2, 12])

def beta_power(signal):
    return psd(signal, fs=256, freq=[12.2, 30])

def gamma_power(signal):
    return psd(signal, fs=256, freq=[30.2, 49])


@jit(nopython=True)
def line_length(signal):
    """
    Measures the line length of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    -------
    line_length : np.float
    """
    # initialize line length
    line_lengths = np.zeros(1)
    line_lengths = line_lengths.astype(np.double)
    
    for i in range(1, signal.shape[0]): # loop though signal
        line_lengths += np.abs(signal[i] - signal[i-1])

    # remove zero-length channels
    line_lengths = line_lengths[np.nonzero(line_lengths)] 
    if line_lengths.size == 0: return 0.0

    # take the median and normalize by clip length
    line_length = np.median(line_lengths) / signal.shape[0]
    return line_length

@jit(nopython=True)
def std(signal):
    """
    Measures standard deviation of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.std(signal)

@jit(nopython=True)
def var(signal):
    """
    Measures variance of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.var(signal)

@jit(nopython=True)
def rms(signal):
    """
    Measures root mean square of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.sqrt(np.mean(np.square(signal)))

@jit(nopython=True)
def max_envelope(signal, win):
    """
    Calculates max envelope of a signal across each window.

    Parameters
    ----------
    signal : 1D numpy array
    win : moving window
    Output
    -------
    max_envelope : 1D numpy array
    """
    env = np.zeros(signal.shape[0])
    for i in range(0,signal.shape[0]): 
        env[i] = np.max(signal[i:i+win])
    return env

@jit(nopython=True)
def get_envelope_max_diff(signal):
    """
    Measures the sum of the differences between the upper and lower envelopes.

    Parameters
    ----------
    signal : 1D numpy array
    win : moving window
    Output
    -------
    max_envelope : 1D numpy array
    """
    # fixed parameter
    win = 30 # moving window (0.3 seconds based on fs=100)
    up_env = max_envelope(signal, win)
    low_env = -max_envelope(-signal, win)  
    return np.sum(up_env - low_env)
    
@jit(nopython=True)
def mad(signal):
    """
    Measures mean absolute deviation of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.mean(np.abs(signal - np.mean(signal)))

@jit(nopython=True)
def energy(signal):
    """
    Measures the energy of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    -------
    energy : np.float
    """
    return np.sum(np.square(signal))


