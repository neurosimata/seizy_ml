# -*- coding: utf-8 -*-

### ------------------------ IMPORTS ---------------------------- ###
import tables
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
### ------------------------------------------------------------- ###

def generate_noisy_sine_waves(num_samples, num_channels, sampling_rate, frequency=10, noise_level=1):
    """
    Generate noisy sine waves.
    
    Parameters
    ----------
    num_samples : int
        Number of samples in the generated sine waves.
    num_channels : int
        Number of channels to generate (equivalent to the number of columns in the output).
    sampling_rate : int
        The sampling rate in Hz.
    frequency : int or float, optional
        The frequency of the sine waves in Hz (default is 10 Hz).
    noise_level : float, optional
        The amplitude of random noise to be added to the sine waves (default is 1).
    
    Returns
    -------
    time : numpy.ndarray
        The time vector corresponding to the generated sine waves.
    noisy_sine_waves : numpy.ndarray
        The generated noisy sine waves of shape (num_samples, num_channels).
        Each column represents a sine wave with added noise.
    
    Notes
    -----
    This function generates noisy sine waves with the specified parameters. It creates `num_channels` sine waves
    with a frequency of `frequency` Hz. The sine waves are sampled at a rate of `sampling_rate` Hz to create
    `num_samples` data points.
    
    Random noise is added to each sine wave with an amplitude defined by `noise_level`. The noise makes the
    generated signals look more biological and realistic.
    
    Example
    -------
    >>> time, data = generate_noisy_sine_waves(num_samples=1000, num_channels=5, sampling_rate=1000, frequency=5, noise_level=0.2)
    """
    time = np.arange(num_samples) / sampling_rate

    # Create the sine waves for each channel
    sine_waves = np.zeros((num_samples, num_channels))
    for channel in range(num_channels):
        sine_waves[:, channel] = np.sin(2 * np.pi * frequency * time)

    # Add some random noise to the sine waves
    noise = np.random.normal(0, noise_level, size=(num_samples, num_channels))
    noisy_sine_waves = sine_waves + noise

    return time, noisy_sine_waves


# Here are creating a datastore in append mode, so that we can load large files in chuncks, 
# downsample them and then store them. When we finish looping through the file we close the datstore.
# After that we can access our h5 file. 

# =====================
# 1. Basic Settings
# =====================

# Here, we setup the basic configuration parameters for our data processing
file_name = 'converted_data.h5'
fs = 1000  # original sampling rate
new_fs = 100  # new sampling rate after decimation
win = 5  # window size in seconds for data storage
downsample_factor = int(fs/new_fs)  # compute the downsample factor

# =======================
# 2. Channel Configurations
# =======================

# We define how many channels our data will have, and which channels we're interested in
total_channels = 5  # total number of channels in the generated data
select_channels = [1, 4]  # list of channels to be selected for decimation

# ==========================
# 3. Datastore Shape Settings
# ==========================

# Here we set up the shape of our datastore.
chunksize = 10000  # number of rows with a bin size of (win*new_fs). **This would define how much data will be loaded.***
cols = int(win*new_fs)  # number of columns in each data row
chunkshape = [chunksize, cols, len(select_channels)]  # shape of each chunk for data storage
datastore_shape = [0, cols, len(select_channels)]  # shape of the complete data

# ==========================
# 4. Create HDF5 Datastore
# ==========================

# Next, we create a HDF5 datastore to save our data.
fsave = tables.open_file(file_name, mode='w')  # open an HDF5 file in write mode
ds = fsave.create_earray(fsave.root, 'data', tables.Float64Atom(),
                         datastore_shape, chunkshape=chunkshape)  # create an extendable array (earray) for data storage

# ==========================
# 5. Generate and Load Data
# ==========================

# Here we generate and load data. In a real use case, this would be replaced by actual data loading
_, data_chunk = generate_noisy_sine_waves(chunksize*fs, total_channels, fs, frequency=10, noise_level=.5)

# ==========================
# 6. Decimate the Data
# ==========================

# We decimate the data from the selected channels
decimate_data = []
for i in select_channels:
    data_ch = signal.decimate(data_chunk[:, i], downsample_factor)
    decimate_data.append(data_ch)
decimate_data = np.vstack(decimate_data).T

# ==========================
# 7. Reshape and Store the Data
# ==========================

# Finally, we reshape the decimated data and append it to our HDF5 datastore
data = decimate_data.reshape(-1, cols, len(select_channels))  # reshape the data to fit the storage shape
ds.append(data)  # append the reshaped data to the HDF5 datastore

# ==========================
# 8. Close the Datastore
# ==========================
fsave.close()


# ------------------------------------------------------------------------- #

# ====================================
# 1. Load Data from the HDF5 Datastore
# ====================================

# Here we open the HDF5 file in read mode, and then load all data into memory
f = tables.open_file(file_name, mode = 'r') # open tables object
load_data = f.root.data[:] # load data
f.close() # always remember to close the file after you're done with it

# =======================================
# 2. Prepare Data for Visualization
# =======================================

# We select a portion of the data that we're interested in visualizing
# Here we select data from the first channel (0) and the first window size 
y = data_chunk[0:(win*fs), select_channels[0]]
y2 = load_data[0,:,0]

# Prepare the time axis for both original and downsampled data
t = np.arange(0,len(y),1) # time axis for the original data
t2 = np.arange(0,len(y),10) # time axis for the downsampled data

# =======================================
# 3. Data Visualization
# =======================================

# Finally, we plot our original data (grey line) and the data we loaded from the HDF5 datastore (black line)
plt.plot(t, y, label='saved_data', linewidth=1, color='grey')
plt.plot(t2, y2, label='loaded_data', linewidth=2, color='black')
plt.legend()




