# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import os
import click
from pathlib import Path
from seizyml.default_settings import (
    get_core_settings,
    get_user_settings,
    CustomOrderGroup,
    core_settings_path,
    user_settings_file,
    save_settings
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
    ctx.obj={}
    core_settings = get_core_settings() 
    user_settings = get_user_settings(Path(core_settings['user_settings_path']))
    ctx.obj['core_settings'] = core_settings
    ctx.obj['user_settings'] = user_settings

# -------------------- MODEL HANDLING -------------------- #
@cli.command(name='train-model')
@click.option('--process', type=click.Choice(['compute_features', 'train_model'], case_sensitive=False), help='Choose to compute features, train model, or none for both.')
@click.pass_context
def train_models(ctx, process):
    """
    1: Train a new seizure detection model.
    """

    # Unpack settings
    usr, core = ctx.obj['user_settings'], ctx.obj['core_settings']

    # Check if model and model path exists
    if core['model_path'] and os.path.isfile(core['model_path']):
        overwrite = click.prompt(f"A model is already trained. Training will overwrite the model used. Proceed (y/n)?\n")
        if overwrite == 'n':
            click.secho("\n -> Operation cancelled. Training aborted.\n", fg='yellow')
            return
        elif overwrite != 'y':
            click.secho("\n -> Invalid input. Please enter 'y' for yes or 'n' for no.\n", fg='red')
            return
        
    # Get training path
    train_path = click.prompt("📂 Enter the training data path", type=str)
    if not os.path.exists(train_path):
        click.secho(f"❌ Train path '{train_path}' not found.", fg='red')
        return

    # Launch the GUI settings editor before training
    from seizyml.user_gui.settings_editor import edit_settings_gui
    click.echo("Opening settings editor GUI for updating settings...")
    updated_settings = edit_settings_gui(usr)
    usr = updated_settings
    click.echo("Settings updated via GUI.")
    
    # Train model and save settings
    from seizyml.train.train_models import train_model
    model_path = train_model(usr, train_path, process)
    core['model_path'] = model_path
    core['user_settings_path'] = os.path.join(train_path, user_settings_file)
    save_settings(core_settings_path, core)
    save_settings(core['user_settings_path'], usr)
    click.secho(f"✅ Created 'user settings' at {core['user_settings_path']}", fg='green')

@cli.command(name='select-model')
@click.argument('model_path')
@click.argument('user_settings_path')
@click.pass_context
def select_model(ctx, model_path, user_settings_path):
    """
    2: Select model trained model and user settings.
    """

    # Validate model file
    try:
        from joblib import load
        load(model_path)
        click.secho(f"✅ Model loaded successfully from '{model_path}'", fg='green')
    except Exception as e:
        click.secho(f"❌ Failed to load model: {e}", fg='red')
        return

    # Validate user settings file loading (The function checks internally)
    get_user_settings(Path(user_settings_path))
    click.secho(f"✅ Settings loaded successfully from '{user_settings_path}'", fg='green')

    # Update core settings
    core =  ctx.obj['core_settings']
    core['model_path'] = model_path
    core['user_settings_path'] = user_settings_path
    save_settings(core_settings_path, core)

@cli.command(name='set-datapath')
@click.argument('path')
@click.pass_context
def set_data_path(ctx, path):
    """
    3: Set the path to the dataset for analysis.
    """

    # Unpack settings
    core = ctx.obj['core_settings']
    usr = ctx.obj['user_settings']

    # Ensure a model is selected before setting the data path
    if not os.path.exists(core.get('model_path')):
        click.secho("❌ A model must be selected before setting the data path. Use 'select-model' first.", fg='red')
        return

    # Validate the provided path
    if not os.path.exists(path):
        click.secho(f"❌ The provided path '{path}' does not exist.", fg='red')
        return

    from seizyml.data_preparation.file_check import validate_data_structure
    # Data Validation
    click.secho(f"🔍 Validating data structure in '{path}'...", fg='cyan')
    data_validation_results = validate_data_structure(
        parent_path=path,
        data_dir=usr['data_dir'],
        model_channels=usr['channels'],
        model_fs=usr['fs'],
        model_win=usr['win']
    )
    if data_validation_results['errors']:
        for error in data_validation_results['errors']:
            click.secho(f"❌ {error}", fg='red')
        click.secho("❗ Data validation failed. Please fix the issues and try again.", fg='yellow')
        return
    else:
        click.secho("✅ Data structure validation successful!", fg='green')

    # Check processing status
    click.secho("📊 Checking data processing status...", fg='cyan')
    # run check for processed and model predictions
    from seizyml.data_preparation.file_check import check_processing_status
    checks = check_processing_status(path, 
                                    usr['data_dir'], 
                                    usr['processed_dir'], 
                                    usr['model_predictions_dir'])
    
    # update core settings
    core['parent_path'] = path
    core['data_validated'] = True
    core['is_processed'] = checks['is_processed']
    core['is_predicted'] = checks['is_predicted']
    save_settings(core_settings_path, core)
    
@cli.command(name='preprocess')
@click.pass_context
def preprocess(ctx):
    """
    4: Preprocess Data: Apply filtering and outlier removal.
    """
    # Unpack settings
    usr, core = ctx.obj['user_settings'], ctx.obj['core_settings']

    # Check if file validation has passed
    if not core.get('data_validated'):
        click.secho("❌ File validation has not passed. Please run `set-datapath` to validate data.", fg='red')
        return

    # Check if data has already been processed
    if core.get('is_processed', False):
        overwrite = click.prompt("⚠️ Files have already been processed. Do you want to overwrite? (y/n)", default='n')
        if overwrite.lower() != 'y':
            click.secho("🚫 Operation cancelled. Existing processed data will remain unchanged.", fg='yellow')
            return

    # Preprocessing paths
    load_path = os.path.join(core['parent_path'], usr['data_dir'])
    save_path = os.path.join(core['parent_path'], usr['processed_dir'])

    # Preprocess Data
    click.secho("⚙️ Preprocessing data: Filtering and removing large outliers...", fg='cyan')
    from seizyml.data_preparation.preprocess import PreProcess
    process_obj = PreProcess(load_path=load_path, save_path=save_path, fs=usr['fs'])
    process_obj.filter_data()

    # Update settings after successful processing
    core['is_processed'] = True
    save_settings(core_settings_path, core)
    click.secho("✅ Data preprocessing completed successfully!", fg='green')

@cli.command(name='predict')
@click.pass_context
def predict(ctx):
    """
    5: Generate Model Predictions.
    """

    # Unpack settings
    usr, core = ctx.obj['user_settings'], ctx.obj['core_settings']

    # Check if data has been preprocessed
    if not core.get('is_processed', False):
        click.secho("❌ Data needs to be preprocessed first. Please run `preprocess`.", fg='red')
        return

    # Check if a trained model is available
    if not core['model_path'] or not os.path.isfile(core['model_path']):
        click.secho(f"❌ No trained model found at {core['model_path']}. Please train or select a model first.", fg='red')
        return
    click.secho(f"📈 Using the following model for predictions: {core['model_path']}", fg='green')

    # Perform Predictions
    click.secho("🚀 Generating model predictions...", fg='cyan')
    from seizyml.data_preparation.get_predictions import ModelPredict
    model_obj = ModelPredict(
        model_path=core['model_path'],
        load_path=os.path.join(core['parent_path'], usr['processed_dir']),
        save_path=os.path.join(core['parent_path'], usr['model_predictions_dir']),
        channels=usr['channels'], win=usr['win'], fs=usr['fs'],
        post_processing_method=usr['post_processing_method'],
        dilation=usr['dilation'],
        erosion=usr['erosion'],
        event_threshold=usr['event_threshold'],
        boundary_threshold=usr['boundary_threshold'],
        rolling_window=usr['rolling_window']
    )
    model_obj.predict()

    # Update core settings after successful predictions
    core['is_predicted'] = True
    save_settings(core_settings_path, core)
    click.secho("✅ Model predictions generated successfully!", fg='green')

@cli.command(name='verify')
@click.pass_context
def verify(ctx):
    """
    6: Verify Detected Seizures.
    """

    # Unpack settings
    usr, core = ctx.obj['user_settings'], ctx.obj['core_settings']

    # Check if model predictions have been generated
    if not core.get('is_predicted', False):
        click.secho("❌ Model predictions have not been generated. Please run `predict` first.", fg='red')
        return

    # Validate that processed data and predictions exist
    click.secho("🔍 Validating predictions and processed data...", fg='cyan')
    import numpy as np
    from seizyml.data_preparation.file_check import check_verified
    verification_status = check_verified(
        folder=core['parent_path'],
        data_dir=usr['processed_dir'],
        csv_dir=usr['model_predictions_dir']
    )

    if verification_status:
        click.secho(f"❌ Verification failed. Missing data in: {verification_status}", fg='red')
        return

    # Launch User Verification GUI
    click.secho("🖥️ Launching seizure verification GUI...", fg='cyan')
    from seizyml.user_gui.user_verify import UserVerify
    verifier = UserVerify(
        parent_path=core['parent_path'],
        processed_dir=usr['processed_dir'],
        model_predictions_dir=usr['model_predictions_dir'],
        verified_predictions_dir=usr['verified_predictions_dir']
    )

    # User selects a file to verify
    file_id = verifier.select_file()

    # Retrieve data, seizure indices, and any existing verification colors
    data, idx_bounds = verifier.get_bounds(file_id, verified=False)
    color_array = None
    verified_path = os.path.join(verifier.verified_predictions_path, file_id)

    if os.path.exists(verified_path):
        try:
            color_array = np.loadtxt(os.path.join(verifier.verified_predictions_path, f'color_{file_id.replace(".csv", ".txt")}'), dtype=str)
        except Exception as e:
            click.secho(f"⚠️ Warning: Failed to load color array for {file_id}: {e}", fg='yellow')

    # Handle case with zero seizures
    if idx_bounds.shape[0] == 0:
        verifier.save_emptyidx(data.shape[0], file_id)
        click.secho(f"ℹ️ No seizures detected in {file_id}. Verification saved as empty.", fg='yellow')
    else:
        from seizyml.user_gui.verify_gui import VerifyGui
        VerifyGui(core['parent_path'], usr, file_id, data, idx_bounds, color_array)
        click.secho("✅ Verification process completed successfully!", fg='green')

@cli.command(name='extract-properties')
@click.pass_context
def extract_properties(ctx):
    """
    6: Extract Seizure Properties.
    """
    # Unpack settings
    usr, core = ctx.obj['user_settings'], ctx.obj['core_settings']

    # Check if verified predictions are available
    ver_path = os.path.join(core['parent_path'], usr['verified_predictions_dir'])
    if not os.path.exists(ver_path):
        click.secho("❌ Verified predictions not found. Please verify detected seizures first.", fg='red')
        return

    filelist = [f for f in os.listdir(ver_path) if f.endswith('.csv')]
    if not filelist:
        click.secho("❌ No verified seizures found. Please verify detected seizures.", fg='red')
        return

    # Extract properties
    click.secho("🔍 Extracting seizure properties...", fg='cyan')
    from seizyml.helper.get_seizure_properties import get_seizure_prop
    _, save_path = get_seizure_prop(core['parent_path'], usr['verified_predictions_dir'], usr['gui_win'])

    click.secho(f"✅ Seizure properties saved in '{save_path}'", fg='green')

@cli.command(name='feature-contribution')
@click.pass_context
def feature_contribution(ctx):
    """
    7: Plot Feature Contributions.
    """
    # Unpack settings
    core = ctx.obj['core_settings']

    # Check if model is trained
    if not core['model_path'] or not os.path.isfile(core['model_path']):
        click.secho(f"❌ Model not found at {core['model_path']}. Please train a model first.", fg='red')
        return

    # Load model and plot feature contributions
    click.secho("📊 Plotting feature contributions...", fg='cyan')
    from joblib import load
    import numpy as np
    import matplotlib.pyplot as plt

    model = load(core['model_path'])
    importances = np.abs(model.theta_[0] - model.theta_[1]) / (np.sqrt(model.var_[0]) + np.sqrt(model.var_[1]))
    importances = importances / np.sum(importances)

    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    idx = np.argsort(importances)
    ax.barh(np.array(model.feature_labels)[idx], importances[idx], facecolor='#66bd7d', edgecolor='#757575')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Feature Separation Score')
    ax.set_ylabel('Features')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    cli(obj={})
