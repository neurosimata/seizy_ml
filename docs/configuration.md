## Configuration settings

All settings are stored in the `config.yaml` file.
** The `config.yaml` is created when the path is first set in **SeizyML** set from [temp_config.yaml](/temp_config.yaml).

#### User Settings

1) Basic data parameters
- **channels** : List containing the names of LFP/EEG channels, e.g. ["hippocampus", "frontal cortex"].
- **win** : Window size in seconds, default and recommended is 5 seconds.
- **fs** : Sampling rate of .h5 files, default and recommended is 100 Hz.

2) Paths
- **parent_path** : Path to parent directory, e.g. "C:\\Users\\...\\parent directory".
- **data_dir** : Child directory name where .h5 files are present, default is **"data"**.
- **processed_dir** : Child directory name with h5 preprocessed data, default is **"processed"**.
- **model_predictions_dir** : Child directory name with model predictions are present (.csv), default is **"model_predictions"**.
- **verified_predictions_dir** : Child directory name where user verified predictions are present (.csv), default is **"verified_predictions"**.

## Path organization

<img src="configuration_paths.png" width="500">

3) Feature selection parameters
- **features** : List containing features to be used for feature selection
- **feature_select_thresh** : Threshold for removing highlly correlated features, defulat is 0.9
- **feature_size** : List containing size of feature sets, default is [5,10,15].
- **nleast_corr** : Number of list correlated features to include.

4) Post processing methods
- **post_processing_method** : Type of post processing method. Options are `dual_threshold` (default), `dilation_erosion`, and `erosion_dilation`.
- **rolling_window** : Window size for smoothing predictions in `dual_threshold`. Default is 6. Higher number increases stringency.
- **event_threshold** : Probability threshold for event detection in `dual_threshold`. Default is .5. Higher number increases stringency.
- **boundary_threshold** : Probability threshold for boundary detection detection in `dual_threshold`. Default is .2. Higher number increases stringency. Has to be lower than event threshold.
- **dilation** : Size of dilation structuring element in `dilation_erosion`, and `erosion_dilation` methods. Default is 2. Higher number reduces stringency.
- **erosion** : Size of erosion structuring element in `dilation_erosion`, and `erosion_dilation` methods. Default is 2. Higher number increases stringency.

5) Used by the app for file check (not to be edited by the user)
   
**[<< Back to Main Page](/README.md)**
