# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import os
import click
import yaml
from seizyml.default_settings import (
    get_core_settings,
    get_user_settings,
    CustomOrderGroup,
    core_settings_path
)
### ----------------------------------------------------------------- ###

@click.group(cls=CustomOrderGroup)
@click.pass_context
def cli(ctx):
    """
    -------------------------------------------------------------
    
    \b                                                             
    \b                       _          ___  ___ _     
    \b                      (_)         |  \/  || |    
    \b              ___  ___  ______   _| .  . || |    
    \b             / __|/ _ \ |_  / | | | |\/| || |    
    \b             \__ \  __/ |/ /| |_| | |  | || |____
    \b             |___/\___|_/___|\__, \_|  |_/\_____/
    \b                             __/ |              
    \b                            |___/                                                  
    \b 

    --------------------------------------------------------------
                                                                                                                                           
    """

    core_settings = get_core_settings()
    ctx.obj = core_settings

    if core_settings.get("default_model"):
        user_settings = get_user_settings(core_settings.get("default_model"))
        ctx.obj.update(user_settings)

# -------------------- MODEL HANDLING -------------------- #
@cli.command(name='train-model')
@click.option('--process', type=click.Choice(['compute_features', 'train_model'], case_sensitive=False), help='Choose to compute features, train model, or both.')
@click.pass_context
def train_model(ctx, process):
    """
    1: Train a new seizure detection model.
    """
    click.echo("🚀 Training model...")

    # check if model and model path exists
    if ctx.obj['model_path'] and not os.path.isfile((ctx.obj['model_path'])):
        click.secho(f"\n -> The the model file cannot be located at {ctx.obj['model_path']}\
                    Consider training a new model or moving the file back to the original location.", fg='yellow', bold=True)

    elif ctx.obj['model_path'] and os.path.isfile((ctx.obj['model_path'])):
        overwrite = click.prompt(f"A model is already trained. Training  will overwrite the model used. Proceed (y/n)?\n")
        if overwrite == 'n':
            click.secho("\n -> Operation cancelled. Training aborted.\n", fg='yellow', bold=True)
            return
        elif overwrite != 'y':
            click.secho("\n -> Invalid input. Please enter 'y' for yes or 'n' for no.\n", fg='red', bold=True)
            return
    
    from seizyml.train.train_models import train_model
    train_model(ctx, process)

@cli.command(name='select-model')
@click.argument('model_path')
@click.pass_context
def select_model(ctx, model_path):
    """
    2: Select an existing trained model.
    """
    core_settings = get_core_settings()
    core_settings['default_model'] = model_path
    with open(core_settings_path, 'w') as file:
        yaml.dump(core_settings, file)
    click.secho(f"✅ Model selected: {model_path}", fg='green')

# -------------------- DATASET CONFIGURATION -------------------- #
@cli.command(name='set-data-path')
@click.argument('path')
@click.pass_context
def set_data_path(ctx, path):
    """
    3: Set the path to the dataset for analysis.
    """
    core_settings = get_core_settings()
    core_settings['parent_path'] = path
    with open(core_settings_path, 'w') as file:
        yaml.dump(core_settings, file)
    click.secho(f"✅ Data path set to: {path}", fg='green')

    # Run file and validation checks
    validation_passed = check_files_and_settings(ctx)
    ctx.obj.update({'data_validated': validation_passed})

    if validation_passed:
        click.secho("✅ Data validation passed.", fg='green')
    else:
        click.secho("❌ Data validation failed. Please fix the issues before proceeding.", fg='red')

@cli.command(name='check-files')
@click.pass_context
def check_files_and_settings(ctx):
    """
    4: Check file structure and validate dataset.
    """
    parent_path = ctx.obj['parent_path']
    data_dir = ctx.obj['data_dir']
    processed_dir = ctx.obj['processed_dir']
    model_predictions_dir = ctx.obj['model_predictions_dir']

    # File structure check
    from seizyml.data_preparation.file_check import check_main
    processed_check, model_predictions_check = check_main(
        parent_path, data_dir, processed_dir, model_predictions_dir
    )

    # Validation check
    data_fs = ctx.obj.get('fs')
    model_fs = ctx.obj.get('fs')
    data_win = ctx.obj.get('win')
    model_win = ctx.obj.get('win')
    data_channels = ctx.obj.get('channels')
    model_channels = ctx.obj.get('channels')

    validation_passed = (data_fs == model_fs) and (data_win == model_win) and (data_channels == model_channels)

    # Update settings
    ctx.obj.update({
        'file_check': processed_check,
        'processed_check': processed_check,
        'predicted_check': model_predictions_check,
        'data_validated': validation_passed
    })

    # Save settings
    with open(ctx.obj['settings_path'], 'w') as file:
        yaml.dump(ctx.obj, file)

    return validation_passed
if __name__ == '__main__':
    cli(obj={})

# @main.command()
# @click.pass_context
# def setpath(ctx):
#     """
#     1: Set path
#     """

#     # get parent path
#     parent_path = click.prompt(f'--> Please enter the Parent path: e.g. users/seizure_data_for_scoring\n')
#     if os.path.exists(parent_path):
#         with open(core_settings_path, 'w') as file:
#             yaml.dump({'data_path':parent_path, 'model_path':ctx.obj['model_path']}, file)
#         click.secho(f"\n -> Parent path was set to:'{parent_path}'.\n", fg='green', bold=True)
#     else:
#         click.secho(f"\n -> Directory not found'.\n", fg='yellow', bold=True)
#         return

#     # Check if the settings file exists
#     if os.path.isfile(os.path.join(parent_path, settings_file_name)):
#         overwrite = input(f'--> A settings file already exists. Do you want to overwrite it? (y/n): ').strip().lower()
#         if overwrite == 'n':
#             click.secho("\n -> Operation cancelled. Existing settings file was not modified.\n", fg='yellow', bold=True)
#             return
#         elif overwrite != 'y':
#             click.secho("\n -> Invalid input. Please enter 'y' for yes or 'n' for no.\n", fg='red', bold=True)
#             return

#     # get parent path and set checks to False
#     settings_path = os.path.join(parent_path, settings_file_name)
#     ctx.obj.update({'settings_path': settings_path,
#                     'parent_path': parent_path,
#                     'file_check':False,
#                     'processed_check':False,
#                     'predicted_check':False,
#                     })
    
#     # run check for processed and model predictions
#     from seizyml.data_preparation.file_check import check_main
#     processed_check, model_predictions_check = check_main(ctx.obj['parent_path'], 
#                                                           ctx.obj['data_dir'], 
#                                                           ctx.obj['processed_dir'], 
#                                                           ctx.obj['model_predictions_dir'])
#     if processed_check:
#             ctx.obj.update({'file_check':True})
#             ctx.obj.update({'processed_check':True})
#     if processed_check and model_predictions_check:
#         ctx.obj.update({'predicted_check':True})
    
#     # create yaml file
#     with open(settings_path, 'w') as file:
#         yaml.dump(ctx.obj, file)
#     click.secho(f"\n -> A settings file was created at:'{settings_path}'.\n", fg='green', bold=True)

# @main.command()
# @click.pass_context
# def filecheck(ctx):
#     """2: Check files"""

#     # get child folders and create success list for each folder
#     if not os.path.exists(ctx.obj['parent_path']):
#         click.secho(f"\n -> Parent path '{ctx.obj['parent_path']}' was not found." +\
#                     " Please run -setpath-.\n", fg='yellow', bold=True)
#         return
    
#     # get channel_names
#     overwrite = click.prompt(f"Got ({', '.join(ctx.obj['channels'])}) as channel names. Do you want to proceed (y/n)?\n")
#     if overwrite == 'n':
#         click.secho("\n -> Operation cancelled. Files were not checked.\n", fg='yellow', bold=True)
#         return
#     elif overwrite != 'y':
#         click.secho("\n -> Invalid input. Please enter 'y' for yes or 'n' for no.\n", fg='red', bold=True)
#         return
    
#     ### code to check for files ###
#     from seizyml.data_preparation.file_check import check_h5_files
#     error = check_h5_files(os.path.join(ctx.obj['parent_path'], ctx.obj['data_dir']),
#                            win=ctx.obj['win'], fs=ctx.obj['fs'], 
#                            channels=len(ctx.obj['channels']))
    
#     if error:
#         click.secho(f"-> File check did not pass {error}\n", fg='yellow', bold=True)
        
#     else:
#         # save error check to settings file
#         ctx.obj.update({'file_check': True})
#         with open(ctx.obj['settings_path'], 'w') as file:
#             yaml.dump(ctx.obj, file) 
#         click.secho(f"\n -> Error check for '{ctx.obj['parent_path']}' has been successfully completed.\n",
#                     fg='green', bold=True)

# @main.command()
# @click.pass_context
# def preprocess(ctx):
#     """3: Pre-process data (filter and remove large outliers) """
    
#     if not ctx.obj['file_check']:
#         click.secho("\n -> File check has not pass. Please run -filecheck-.\n", fg='yellow', bold=True)
#         return
    
#     if ctx.obj['processed_check'] == True:
#         overwrite = click.prompt(f"Files have already been processed. Do you want to proceed (y/n)?\n")
#         if overwrite == 'n':
#             click.secho("\n -> Operation cancelled. Files will not be processed.\n", fg='yellow', bold=True)
#             return
#         elif overwrite != 'y':
#             click.secho("\n -> Invalid input. Please enter 'y' for yes or 'n' for no.\n", fg='red', bold=True)
#             return
        
#     from seizyml.data_preparation.preprocess import PreProcess
#     # get paths, preprocess and save data
#     load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['data_dir'])
#     save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
#     process_obj = PreProcess(load_path=load_path, save_path=save_path, fs=ctx.obj['fs'])
#     process_obj.filter_data()
#     ctx.obj.update({'processed_check':True})
#     click.secho("\n -> File preprocessing completed successfully.\n", fg='green', bold=True)
#     with open(ctx.obj['settings_path'], 'w') as file:
#         yaml.dump(ctx.obj, file) 
#     return
 
# @main.command()
# @click.pass_context
# def predict(ctx):
#     """4: Generate model predictions"""
    
#     # checks if files have been preprocessed and a model was trained
#     if ctx.obj['processed_check'] == False:
#         click.secho("\n -> Data need to be preprocessed first. Please run -preprocess-.\n",
#                     fg='yellow', bold=True)
#         return
    
#     # check if model was trained
#     if not os.path.isfile((ctx.obj['model_path'])):
#         click.secho(f"Model {ctx.obj['model_path']} could not be found. Please train a model.", fg='yellow', bold=True)
#         return
#     else:
#         click.secho(f"\n ->The following model will be used for predictions: {ctx.obj['model_path']}.\n", fg='green', bold=True)
    
#     from seizyml.data_preparation.get_predictions import ModelPredict
#     # get paths and model predictions
#     load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
#     save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['model_predictions_dir'])
#     model_obj = ModelPredict(ctx.obj['model_path'], load_path, save_path, 
#                              channels=ctx.obj['channels'], win=ctx.obj['win'], fs=ctx.obj['fs'],
#                              post_processing_method=ctx.obj['post_processing_method'], dilation=ctx.obj['dilation'],
#                              erosion=ctx.obj['erosion'], event_threshold=ctx.obj['event_threshold'], 
#                              boundary_threshold=ctx.obj['boundary_threshold'], rolling_window=ctx.obj['rolling_window'],)
#     model_obj.predict()
#     ctx.obj.update({'predicted_check':True})
    
#     with open(ctx.obj['settings_path'], 'w') as file:
#         yaml.dump(ctx.obj, file)
#     return

# @main.command()
# @click.pass_context
# def verify(ctx):
#     """5: Verify detected seizures"""
    
#     if ctx.obj['predicted_check'] == False:
#         click.secho("\n -> Model predictions have not been generated. Please run -predict-.\n",
#                     fg='yellow', bold=True)
#         return
    
#     import numpy as np
#     from seizyml.data_preparation.file_check import check_verified
#     out = check_verified(folder=ctx.obj['parent_path'],
#                      data_dir=ctx.obj['processed_dir'],
#                      csv_dir=ctx.obj['model_predictions_dir'])
#     if out:
#         click.secho(f"\n -> Error. Could not find: {out}.\n",
#              fg='yellow', bold=True)
#         return
    
#     # Create instance for UserVerify class
#     from seizyml.user_gui.user_verify import UserVerify
#     obj = UserVerify(ctx.obj['parent_path'],
#                      ctx.obj['processed_dir'], 
#                      ctx.obj['model_predictions_dir'],
#                      ctx.obj['verified_predictions_dir'])
    
#     # user file selection
#     file_id = obj.select_file()
                  
#     # check if file was verified and get data, seizure index, and color array (if verified)
#     data, idx_bounds = obj.get_bounds(file_id, verified=False)
#     if os.path.exists(os.path.join(obj.verified_predictions_path, file_id)):
#         try:
#             color_array = np.loadtxt(os.path.join(obj.verified_predictions_path, 'color_' + file_id.replace('.csv', '.txt')), dtype=str)
#         except:
#             color_array = None
#     else:
#         color_array = None
        
#     # check for zero seizures otherwise proceed with gui creation
#     if idx_bounds.shape[0] == 0:
#         obj.save_emptyidx(data.shape[0], file_id)     
#     else:
#         from seizyml.user_gui.verify_gui import VerifyGui
#         VerifyGui(ctx.obj, file_id, data, idx_bounds, color_array)

# @main.command()
# @click.pass_context
# def extractproperties(ctx):
#     """6: Get seizure properties"""
    
#     ver_path = os.path.join(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'])
#     if os.path.exists(ver_path):
#         filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))

#     if not filelist:
#         click.secho("\n -> Could not find verified seizures: Please verify detected seizures.\n",
#              fg='yellow', bold=True)
#         return
    
#     # get properies and save
#     from seizyml.helper.get_seizure_properties import get_seizure_prop
#     _, save_path = get_seizure_prop(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'], ctx.obj['gui_win'])
#     click.secho(f"\n -> Properies were saved in '{save_path}'.\n", fg='green', bold=True)

# @main.command()
# @click.pass_context
# def featurecontribution(ctx):
#     """7: Plot feature contibutions"""
    
#     # check if model was trained
#     if not os.path.isfile((ctx.obj['model_path'])):
#         click.secho(f"Model {ctx.obj['model_path']} could not be found. Please train a model.", fg='yellow', bold=True)
#         return
    
#     from joblib import load
#     import numpy as np
#     import matplotlib.pyplot as plt
#     model_path = ctx.obj['model_path']
#     model = load(model_path)
#     importances = np.abs(model.theta_[0] - model.theta_[1]) / (np.sqrt(model.var_[0]) + np.sqrt(model.var_[1]))
#     importances = importances/np.sum(importances)
#     plt.figure(figsize=(5,3))
#     ax = plt.axes()
#     idx = np.argsort(importances)
#     ax.barh(np.array(model.feature_labels)[idx], importances[idx], facecolor='#66bd7d', edgecolor='#757575')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_xlabel('Feature Separation Score')
#     ax.set_ylabel('Features')
#     plt.tight_layout()
#     plt.show()




