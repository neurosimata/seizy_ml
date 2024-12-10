# SeizyML
:snake: **SeizyML** uses interpretable machine learning models to detect :detective: seizures from EEG recordings.
After the seizures are detected, they can be manually verified with a user-friendly GUI.

---
## Summary
- [Installation](#installation)
- [Launch App](#launch-app)
- [App Configuration](#app-configuration)
- [Model Training](#model-training)
- [How To Use](#how-to-use)

---

### Installation
1) Download and install [miniconda](https://repo.anaconda.com/miniconda/) on your platform.
2) Clone or Download [SeizyML](https://github.com/neurosimata/seizy_ml/)
3) Start Anaconda's shell prompt, navigate to */seizy_ml* and create conda environment:

        conda env create -f environment.yml
        
---

### Launch App

Via Anaconda's shell prompt

        # navigate to *seizy* folder
        cd ./seizy_ml
        
        # enter conda environment
        conda activate seizyml

        # Launch CLI
        python cli.py

If this works it should display the SeizyMl cli App.

<img src="docs/cli.png" width="500">
---  

### App Configuration

All settings are stored in the `config.yaml` file. 
- This file will be created in the **SeizyML** folder from a template file (`temp_config.yaml`) after you use the setpath command for the first time. ⚠️ The `temp_config.yaml` file should not be edited by the user. 

To edit the `config.yaml` use any text editor such as notepad:
- The only setting that requires editing before training a model and using the app is the `channels` field.
-         **channels** : List containing the names of LFP/EEG channels, e.g. ["hippocampus", "frontal cortex"]
- All other settings can be left at default, given that the data were prepared in the recommended format (.h5 files with shape **[Nsegments, 500 (1 segment), channels]**).
- For data conversion check the accompanying app [seizy_convert](https://github.com/neurosimata/seizy_convert) or the [h5_conversion script](/examples/to_h5.py) for more help.
- An explanation of all other settings can be found [here](/docs/configuration.md).
---
        
### Model Training
- Before using **SeizyML** for seizure detection a model should be first trained on ground truth (hand-scored) data.

1) **Launch Conda Shell Prompt, navigate to seizy_ml directory and activate the virtual environment.**
```
cd ./seizy_ml
conda activate seizyml
```

2) **Set path for data processing.**
```
python cli.py setpath 'path'
```
- This is the folder path where the training data in .h5 format along with the corresponding training labels in .csv format are stored.
- The training data consist of each recording in .h5 format **[Nsegments, 1 segment, Nchannels].** Where a segment is 500 (win*fs).
- The training labels consist of a corresponding .csv file containing the  ground truth labels (1 for seizure, 0 for non seizure) with length **[Nsegments].**
- Training data and labels for each recording need to have a matching name.

  <img src="docs/train_files.png" width="500">
  
- The `win`, `fs`, `channels` fields should be set in `config.yaml` to match the shape of the data. Defaults are win=5, fs=100.
-  The `config.yaml` is created when the path is first set in **SeizyML** set from [temp_config.yaml](/temp_config.yaml).
- **This folder** should be kept in **one location** as the trained models will be stored here.
- **If the folder is moved**, then the `training_path` field in `config.yaml` should be **updated** to reflect the new location.

3) **Model Training**
```
python cli.py train
```
- This is a multi-step process:
    - a) Data preprocessing (high pass filter and exterme outlier removal).
    - b) Feature extraction.
    - c) Find six best feature sets.
    - d) Train a GNB model on these feature sets and select the one with highest F1 score.
    - The *model_id* will be stored in the config.yaml file and will be used to load that model.
      
4) **Feature Contributions**
Features contribution to the GNB model can be visualized using the following command.
```
python cli.py featurecontribution
```
---
        
### How To Use

** **Note:** ** A model must be [trained](#model-training) ☝️ before using the app for seizure detection.

1) **Launch Conda Shell Prompt, navigate to seizy_ml directory and activate the virtual environment.**
```
cd ./seizy_ml
conda activate seizyml
```

2) **Set path for data processing.**
```
python cli.py setpath 'path'
```
- This is the parent path where the directory ('data_dir') with h5 data resides [configuration settings](/docs/configuration.md).
- All subsequent folders and model predictions will reside here.

3) **Run file check.**
```
python cli.py filecheck
```
- ⚠️ This step checks that the h5 files have the correct dimensions. For help on how to convert files to h5 have a look at the [h5_conversion script](/examples/to_h5.py).

4) **Preprocess data.**

- This is the step where the h5 data files will be filtered and large outliers will be removed.

```
python cli.py preprocess
```

5) **Generate model prections.**
```
python cli.py predict
```
- Here selected features will be extracted and model predictions will be generated using the selected model (model id can be found in the configuration settings file).

6) **Verify seizures and adjust seizure boundaries.**
- This will launch a prompt to allow for file selection for verification.
- After the file selection, a GUI will be launched for seizure verfication and seizure boundary adjustment. 
```
python cli.py verify
```

<img src="docs/verify_gui.png" width="500">

7) **Get seizure properties.** 
-This step will generate a csv file with seizure properties for each h5 file.
```
python cli.py extractproperties
```

----

### Contributions
We welcome all project contributions including raising issues and pull requests!

----

-> Back to Page [top](#summary).
