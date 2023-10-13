## Training Tutorial

1) **Launch Conda Shell Prompt, navigate to seizy_ml directory and activate the virtual environment.**
```
cd ./seizy_ml
conda activate seizyml
```

2) **Set path for data processing.**
```
python cli.py setpath 'path'
```
- This is the path to h5 training data. 
- ⚠️ The training data should be named `data.h5` and have a shape of **[nrows, 500 (win*fs), channels].**
- ⚠️ The training labels should be named  `y.h5` and have a shape of **[nrows].**
- ⚠️ The `win`, `fs`, `channels` fields should be set in [config.yaml](/config.yaml) to match the shape of the data.
- **This folder** should be kept in **one location** as the trained models will be stored here.
- **If the folder is moved**, then the `training_path` field in [config.yaml](/config.yaml) should be **updated** to reflect the new location.

3) **Train model**
```
python cli.py train
```
- This is a multi-step process:
    - a) Data preprocessing (high pass filter and exterme outlier removal).
    - b) Feature extraction.
    - c) Find 5 best feature sets and save.
    - d) Train a GNB model on these 5 feature sets and select the one with highest F1 score.
    - The *model_id* will be stored in the [config.yaml](/config.yaml) file and will be used to load that model.
      
**[<< Back to Main Page](/README.md)**
