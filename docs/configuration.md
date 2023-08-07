## Configuration settings

#### -> To be provided by user
- **parent_path** : path to parent directory, e.g. "C:\\Users\\...\\parent directory"
- **data_dir** : child directory name where .h5 files are present, default is "h5_data"
- **channels** : List containing the names of LFP/EEG channels, e.g. ["hippocampus", "frontal cortex"]
- **win** : window size in seconds, default and recommended is 5 seconds
- **fs** : sampling rate of .h5 files, default and recommended is 100 Hz

:exclamation:The original LFP/EEG data have to be converted to .h5 files with the following 3D shape **[nrows, 500 (win*fs), 2 (Nchannels)].**
Check out the [h5_conversion script](/examples/to_h5.py) for more help.

---

#### -> Created during app execution
- **processed_dir** : child directory name with h5 preprocessed data, default is "processed"
- **model_predictions_dir** : child directory name with model predictions are present (.csv), default is "model_predictions"
- **verified_predictions_dir** : child directory name where user verified predictions are present (.csv), default is "verified_predictions"

---
## Path organization

<img src="configuration_paths.png" width="500">

---

**[<< Back to Main Page](/README.md)**
