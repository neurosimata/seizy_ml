<p align="center">
        <img src="docs/seizyML.png" width="600">
</p>

- **SeizyML** uses interpretable machine learning models to detect 🕵️‍♂️ seizures from EEG recordings coupled with manual verification in user-friendly GUI.
- 📖 To reference **SeizyML**, or view the manuscript, please refer to the following [publication](https://www.researchsquare.com/article/rs-4361048/v1) (To be updated soon). 
- You can access the data and code used to reproduce the experiments and figures from the accompanying paper on [Zenodo](https://doi.org/10.5281/zenodo.14825785).

![Version](https://img.shields.io/badge/python_version-3.9-purple)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14825785.svg)](https://doi.org/10.5281/zenodo.14825785) 
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)


---
## 📚 Contents
- [⚙️ Hardware Requirements](#hardware-requirements)
- 💾 [Installation](#installation)
- 🛠️ [App Configuration](#app-configuration)
- 📈 [Model Training](#model-training)
- 🚀 [How To Use](#how-to-use)

### 📄 Additional Resources
- [Configuration settings](/docs/configuration.md)
- [SeizyML Processing Pipeline](/docs/model_pipeline.md)
- [Custom Model Integration](/docs/custom_model.md)
---

### Hardware requirements

- **SeizyML** is a lightweight application that utilizes Gaussian Naive Bayes (GNB) models to predict seizure events from EEG data.
- Any modern CPU with sufficient RAM to load your EEG recordings should work effectively.
- For example, a **quad-core CPU with 16 GB RAM** can efficiently handle 24-hour long EEG recordings with 2 channels sampled at 4000 Hz.
- **No GPU is required** for SeizyML's operation.
---

### Installation

#### Conda (Recommended)
1) Download and install [miniconda](https://repo.anaconda.com/miniconda/).
2) Clone or Download [SeizyML](https://github.com/neurosimata/seizy_ml/) on your machine.
3) Start Anaconda's prompt, navigate to the downloaded */seizy_ml* to create the conda environment:

        conda env create -f environment.yml

4) Activate environment

        conda activate seizyml

5) Launch App

        seizyml

#### Pip
1) Download and install [Python 3.9](https://www.python.org/downloads/release/python-390/).

2) In the terminal

        pip install seizyml

3) Launch App

        seizyml

If this works you should see the SeizyMl CLI interface.

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

1) **Launch App.**

For conda:
```
# In anaconda prompt
cd ./seizy_ml
conda activate seizyml
seizyml
```

For pip:
```
# In terminal
seizyml
```

2) **Set path for data processing.**
```
seizyml setpath 'path'
```
- This is the folder path where the training data in .h5 format along with the corresponding training labels in .csv format are stored.
- The training data consist of each recording in .h5 format **[Nsegments, 1 segment, Nchannels].** Where a segment is 500 (win*fs).
- The training labels consist of a corresponding .csv file containing the  ground truth labels (1 for seizure, 0 for non seizure) with length **[Nsegments].**
- Training data and labels for each recording need to have a matching name.

  <img src="docs/train_files.png" width="500">
  
- The `win`, `fs`, `channels` fields should be set in `config.yaml` to match the shape of the data. Defaults are win=5, fs=100.
-  The `config.yaml` is created when the path is first set in **SeizyML** set from [temp_config.yaml](seizyml/temp_config.yaml).
- **This folder** should be kept in **one location** as the trained models will be stored here.
- **If the folder is moved**, then the `training_path` field in `config.yaml` should be **updated** to reflect the new location.

3) **Model Training**
```
seizyml train
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
seizyml featurecontribution
```
---
        
### How To Use

⚠️ **Note:** A model must be [trained](#model-training) ☝️ before using the app for seizure detection.

1) **Launch App.**

For conda:
```
# In anaconda prompt
cd ./seizy_ml
conda activate seizyml
seizyml
```

For pip:
```
# In terminal
seizyml
```

2) **Set path for data processing.**
```
seizyml setpath 'path'
```
- This is the parent path where the child folder with h5 data resides and where all subsequent folders will be created. Check [configuration settings](/docs/configuration.md) for more information.
- The h5 data should be added in a child folder called `data`.

3) **Run file check.**
```
seizyml filecheck
```
- ⚠️ This step checks that the h5 files have the correct dimensions. For help on how to convert files to h5 have a look at the [h5_conversion script](/examples/to_h5.py).

4) **Preprocess data.**

- This is the step where the h5 data files will be filtered and large outliers will be removed.

```
seizyml preprocess
```

5) **Generate model prections.**
```
seizyml predict
```
- Here selected features will be extracted and model predictions will be generated using the selected model (model id can be found in the configuration settings file).

6) **Verify seizures and adjust seizure boundaries.**
- This will launch a prompt to allow for file selection for verification.
- After file selection, a GUI will be launched for seizure verification and boundary adjustment.

```
seizyml verify
```

<img src="docs/verify_gui.png" width="500">

7) **Get seizure properties.** 
-This step will generate a csv file with seizure properties for each h5 file.
```
seizyml extractproperties
```

----

### Contributions
We welcome all project contributions including raising issues and pull requests!

----

-> Back to [Top](#contents).
