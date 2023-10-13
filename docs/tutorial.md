## Main Tutorial

1) **Launch Conda Shell Prompt, navigate to seizy_ml directory and activate the virtual environment.**
```
cd ./seizy_ml
conda activate seizyml
```

2) **Set path for data processing.**
```
python cli.py setpath 'path'
```
- This is the parent path where the directory ('data_dir') with h5 data resides [configuration settings](configuration.md).
- All subsequent folders and model predictions will reside here.

3) **Run file check.**
```
python cli.py filecheck
```
- This step checks that the h5 files have the correct dimensions. For help on how to convert files to h5 have a look at the [h5_conversion script](/examples/to_h5.py).
- Each h5 file should contain **Two** LFP/EEG channels [configuration settings](configuration.md).

4) **Preprocess data.**

- This is the step where the h5 data files will be filtered and large outliers will be removed.

```
python cli.py preprocess
```

5) **Generate model prections.**
```
python cli.py predict
```
- Here selected features will be extracted and model predictions will be generated using the selected model from [training](trainining.md).

6) **Verify seizures and adjust seizure boundaries.**
- This will launch a prompt to allow for file selection for verification.
- After the file selection, a GUI will be launched for seizure verfication and seizure boundary adjustment. 
```
python cli.py verify
```

<img src="verify_gui.png" width="500">

7) **Get seizure properties.** 
-This step will generate a csv file with seizure properties for each h5 file.
```
python cli.py extractproperties
```

----

**[<< Back to Main Page](/README.md)**


