# -*- coding: utf-8 -*-

### -------------- IMPORTS -------------- ###
import numpy as np
from scipy.fftpack import fft
import scipy.stats as stats
### ------------------------------------- ###


# =============================================================================
#  Single Features
# =============================================================================

def get_psd(signal, fs):
    """
    Calculate Power spectral Density.

    Parameters
    ----------
    signal : 1D numpy array
    fs: sampling rate

    Returns
    -------
    freqs : 1D numpy array
    psd : 1D numpy array


    """
    # get power spectrum and normalize to signal length
    xdft = np.square(np.absolute(fft(signal)))
    xdft = xdft * (1/(fs*signal.shape[0]))
       
    # multiply *2 to conserve energy in positive frequencies
    psd = 2*xdft[0:int(xdft.shape[0]/2+1)]
    
    # generate corresponding frequency bins
    freqs = np.linspace(0, fs / 2, len(psd))
    return  freqs, psd

def get_power_area(signal, freq, fs):
    """
    Measure the power area of the signal.
    
    Parameters
    ----------
    signal : 1D numpy array
    freq: list(low, high)
    fs: sampling rate
    
    Returns
    -------
    power_area : np.float
    
    """
    
    freqs, psd = get_psd(signal, fs)
    flow = np.argmin(np.abs(freqs - freq[0]))
    fup = np.argmin(np.abs(freqs - freq[1]))
    return np.sum(psd[flow:fup+1])

def weighted_frequency(signal, fs):
    """
    Compute the weighted mean frequency (spectral centroid) of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    fs: sampling rate
    
    Returns
    -------
    wm_frequency : np.float

    """
    freqs, psd = get_psd(signal, fs)
    wm_frequency = np.sum(freqs * psd) / np.sum(psd)
    return wm_frequency

def spectral_entropy(signal, fs):
    """
    Compute the spectral entropy of a normalized power spectral density (PSD).

    Parameters
    ----------
    psd : 1D numpy array
        Normalized power spectral density array.
    fs: sampling rate

    Returns
    -------
    spectral_ent : float
        Spectral entropy value.
    """
    _, psd = get_psd(signal, fs)
    psd_norm = psd / np.sum(psd)
    spectral_ent = stats.entropy(psd_norm)
    return spectral_ent

def delta_power(signal, fs):
    return get_power_area(signal, fs=fs, freq=[2, 4])

def theta_power(signal, fs):
    return get_power_area(signal, fs=fs, freq=[4.2, 8])

def alpha_power(signal, fs):
    return get_power_area(signal, fs=fs, freq=[8.2, 12])

def beta_power(signal, fs):
    return get_power_area(signal, fs=fs, freq=[12.2, 30])

def gamma_power(signal, fs):
    return get_power_area(signal, fs=fs, freq=[30.2, 50])


def line_length(signal, **kwargs):
    """
    Measures the line length of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    -------
    line_length : np.float
    """
    return np.sum(np.abs(np.diff(signal))) / len(signal)

def skewness(signal, **kwargs):
    """
    Compute the skewness of a signal.

    Parameters
    ----------
    signal : 1D numpy array
        Input signal.

    Returns
    -------
    skewness_value : float
        Skewness of the signal.
    """
    return stats.skew(signal)

def kurtosis(signal, **kwargs):
    """
    Compute the kurtosis of a signal.

    Parameters
    ----------
    signal : 1D numpy array
        Input signal.

    Returns
    -------
    kurtosis_value : float
        Kurtosis of the signal.
    """
    return stats.kurtosis(signal)


def var(signal, **kwargs):
    """
    Measures variance of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.var(signal)


def hjorth_mobility(signal, **kwargs):
    """
    Compute the Hjorth Mobility parameter of a signal.

    Parameters
    ----------
    signal : 1D numpy array
        Input signal.

    Returns
    -------
    mobility : float
        Hjorth Mobility of the signal.
    """
    diff_signal = np.diff(signal)
    return np.sqrt(np.var(diff_signal) / np.var(signal))


def hjorth_complexity(signal, **kwargs):
    """
    Compute the Hjorth Complexity parameter of a signal.

    Parameters
    ----------
    signal : 1D numpy array
        Input signal.

    Returns
    -------
    complexity : float
        Hjorth Complexity of the signal.
    """
    diff_signal = np.diff(signal)
    diff_diff_signal = np.diff(diff_signal)
    mobility = np.sqrt(np.var(diff_signal) / np.var(signal))
    complexity = np.sqrt(np.var(diff_diff_signal) / np.var(diff_signal)) / mobility
    return complexity

def rms(signal, **kwargs):
    """
    Measures root mean square of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.sqrt(np.mean(np.square(signal)))

def max_envelope(signal, win, **kwargs):
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
    env[-win + 1:] = env[-win]
    return env

def get_envelope_max_diff(signal, **kwargs):
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
    
def mad(signal, **kwargs):
    """
    Measures median absolute deviation of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.median(np.abs(signal - np.mean(signal)))

def energy(signal, **kwargs):
    """
    Measures the energy of a signal.

    Parameters
    ----------
    signal : 1D numpy array
    -------
    energy : np.float
    """
    return np.sum(np.square(signal))

