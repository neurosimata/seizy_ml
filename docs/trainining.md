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
- All subsequent model predictions will reside here,
- **This folder should be kept in one location as the trained models will be stored here.**
- **If the folder is moved, then the *training_path* in config.yaml should be updated to reflect the new location.**

3) **Train model**
```
python cli.py train
```
- This is a multi-step process:
    - a) Data preprocessing (high pass filter and exterme outlier removal)
    - b) Feature extraction
    - c) Find best 5 feature sets
    - d) Train a GNB model on these 5 feature sets and select the one with highest F1 score
    - The model id will be stored in the config.yaml file and will be used to load that model