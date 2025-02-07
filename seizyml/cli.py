# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import click
import os
import yaml
### ----------------------------------------------------------------- ###

class CustomOrderGroup(click.Group):
    """
    A custom Click group that maintains the order of commands added to it.

    This class extends the `click.Group` class and overrides the `command` and `list_commands` methods
    to keep track of the order in which commands are added to the group. The order is stored in the
    `commands_in_order` attribute, which is a list of command names.

    Attributes:
        commands_in_order (list): A list of command names in the order they were added to the group.
    """
    def __init__(self, **attrs):
        super(CustomOrderGroup, self).__init__(**attrs)
        self.commands_in_order = []

    def command(self, *args, **kwargs):
        def decorator(f):
            cmd = super(CustomOrderGroup, self).command(*args, **kwargs)(f)
            self.commands_in_order.append(cmd.name)
            return cmd
        return decorator

    def list_commands(self, ctx):
        return self.commands_in_order

@click.group(cls=CustomOrderGroup)
@click.pass_context
def main(ctx):
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
        
    # get settings and pass to context
    with open(settings_path, 'r') as file:
        settings = yaml.safe_load(file)
        ctx.obj = settings.copy()
    
@main.command()
@click.argument('path',  type=click.Path())
@click.pass_context
def setpath(ctx, path):
    """
    1: Set path
    **Arguments:path**
    """
    
    # get parent path and set checks to False
    ctx.obj.update({'parent_path': path})
    ctx.obj.update({'file_check':False})
    ctx.obj.update({'processed_check':False})
    ctx.obj.update({'predicted_check':False})
    
    # run check for processed and model predictions
    from seizyml.data_preparation.file_check import check_main
    processed_check, model_predictions_check = check_main(ctx.obj['parent_path'], 
                                                          ctx.obj['data_dir'], 
                                                          ctx.obj['processed_dir'], 
                                                          ctx.obj['model_predictions_dir'])
    if processed_check:
            ctx.obj.update({'file_check':True})
            ctx.obj.update({'processed_check':True})
    if processed_check and model_predictions_check:
        ctx.obj.update({'predicted_check':True})
    
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj, file) 
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg='green', bold=True)
        
@main.command()
@click.pass_context
def filecheck(ctx):
    """2: Check files"""
    
    # get child folders and create success list for each folder
    if not os.path.exists(ctx.obj['parent_path']):
        click.secho(f"\n -> Parent path '{ctx.obj['parent_path']}' was not found." +\
                    " Please run -setpath-.\n",
                    fg='yellow', bold=True)
        return
    
    ### code to check for files ###
    from seizyml.data_preparation.file_check import check_h5_files
    error = check_h5_files(os.path.join(ctx.obj['parent_path'], ctx.obj['data_dir']),
                           win=ctx.obj['win'], fs=ctx.obj['fs'], 
                           channels=len(ctx.obj['channels']))
    
    if error:
        click.secho(f"-> File check did not pass {error}\n", fg='yellow', bold=True)
        
    else:
        # save error check to settings file
        ctx.obj.update({'file_check': True})
        with open(settings_path, 'w') as file:
            yaml.dump(ctx.obj, file) 
        click.secho(f"\n -> Error check for '{ctx.obj['parent_path']}' has been completed.\n",
                    fg='green', bold=True)

@main.command()
@click.pass_context
def preprocess(ctx):
    """3: Pre-process data (filter and remove large outliers) """
    
    if not ctx.obj['file_check']:
        click.secho("\n -> File check has not pass. Please run -filecheck-.\n",
                    fg='yellow', bold=True)
        return
    
    from seizyml.data_preparation.preprocess import PreProcess
    # get paths, preprocess and save data
    load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['data_dir'])
    save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
    process_obj = PreProcess(load_path=load_path, save_path=save_path, fs=ctx.obj['fs'])
    process_obj.filter_data()
    ctx.obj.update({'processed_check':True})

    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj, file) 
    return
 
@main.command()
@click.pass_context
def predict(ctx):
    """4: Generate model predictions"""
    
    if ctx.obj['processed_check'] == False:
        click.secho("\n -> Data need to be preprocessed first. Please run -preprocess-.\n",
                    fg='yellow', bold=True)
        return
    
    from seizyml.data_preparation.get_predictions import ModelPredict
    
    # get paths and model predictions
    model_path = os.path.join(ctx.obj['train_path'], ctx.obj['trained_model_dir'], ctx.obj['model_id'])
    load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
    save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['model_predictions_dir'])
    model_obj = ModelPredict(model_path, load_path, save_path, channels=ctx.obj['channels'], win=ctx.obj['win'], fs=ctx.obj['fs'],
                             post_processing_method=ctx.obj['post_processing_method'], dilation=ctx.obj['dilation'],
                             erosion=ctx.obj['erosion'], event_threshold=ctx.obj['event_threshold'], 
                             boundary_threshold=ctx.obj['boundary_threshold'], rolling_window=ctx.obj['rolling_window'],)
    model_obj.predict()
    ctx.obj.update({'predicted_check':True})
    
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj, file)
    return

@main.command()
@click.pass_context
def verify(ctx):
    """5: Verify detected seizures"""
    
    if ctx.obj['predicted_check'] == False:
        click.secho("\n -> Model predictions have not been generated. Please run -predict-.\n",
                    fg='yellow', bold=True)
        return
    
    import numpy as np
    from seizyml.data_preparation.file_check import check_verified
    out = check_verified(folder=ctx.obj['parent_path'],
                     data_dir=ctx.obj['processed_dir'],
                     csv_dir=ctx.obj['model_predictions_dir'])
    if out:
        click.secho(f"\n -> Error. Could not find: {out}.\n",
             fg='yellow', bold=True)
        return
    
    # Create instance for UserVerify class
    from seizyml.user_gui.user_verify import UserVerify
    obj = UserVerify(ctx.obj['parent_path'],
                     ctx.obj['processed_dir'], 
                     ctx.obj['model_predictions_dir'],
                     ctx.obj['verified_predictions_dir'])
    
    # user file selection
    file_id = obj.select_file()
                  
    # check if file was verified and get data, seizure index, and color array (if verified)
    data, idx_bounds = obj.get_bounds(file_id, verified=False)
    if os.path.exists(os.path.join(obj.verified_predictions_path, file_id)):
        try:
            color_array = np.loadtxt(os.path.join(obj.verified_predictions_path, 'color_' + file_id.replace('.csv', '.txt')), dtype=str)
        except:
            color_array = None
    else:
        color_array = None
        
    # check for zero seizures otherwise proceed with gui creation
    if idx_bounds.shape[0] == 0:
        obj.save_emptyidx(data.shape[0], file_id)     
    else:
        from seizyml.user_gui.verify_gui import VerifyGui
        VerifyGui(ctx.obj, file_id, data, idx_bounds, color_array)

@main.command()
@click.pass_context
def extractproperties(ctx):
    """6: Get seizure properties"""
    
    ver_path = os.path.join(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'])
    if os.path.exists(ver_path):
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))

    if not filelist:
        click.secho("\n -> Could not find verified seizures: Please verify detected seizures.\n",
             fg='yellow', bold=True)
        return
    
    # get properies and save
    from seizyml.helper.get_seizure_properties import get_seizure_prop
    _, save_path = get_seizure_prop(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'], ctx.obj['gui_win'])
    click.secho(f"\n -> Properies were saved in '{save_path}'.\n", fg='green', bold=True)

@main.command()
@click.pass_context
def featurecontribution(ctx):
    """7: Plot feature contibutions"""
    
    # check if model was trained
    if not ctx.obj['model_id']:
        click.secho("No model was found. Please train a model.",
                    fg='yellow', bold=True)
    
    from joblib import load
    import numpy as np
    import matplotlib.pyplot as plt
    model_path = os.path.join(ctx.obj['train_path'], ctx.obj['trained_model_dir'], ctx.obj['model_id']+'.joblib')
    model = load(model_path)
    importances = np.abs(model.theta_[0] - model.theta_[1]) / (np.sqrt(model.var_[0]) + np.sqrt(model.var_[1]))
    importances = importances/np.sum(importances)
    plt.figure(figsize=(5,3))
    ax = plt.axes()
    idx = np.argsort(importances)
    ax.barh(np.array(model.feature_labels)[idx], importances[idx], facecolor='#66bd7d', edgecolor='#757575')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Feature Separation Score')
    ax.set_ylabel('Features')
    plt.tight_layout()
    plt.show()

@main.command()
@click.option('--p', type=str, help='compute_features, train_model')
@click.pass_context
def train(ctx, p):
    """* Train Models """
    
    # get child folders and create success list for each folder
    if not os.path.exists(ctx.obj['parent_path']):
        click.secho(f"\n -> Parent path '{ctx.obj['parent_path']}' was not found." +\
                    " Please run -setpath-.\n",
                    fg='yellow', bold=True)
        return
    
    # imports
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from seizyml.train.train_models import train_and_save_models
    from seizyml.helper.io import load_data, save_data
    from seizyml.data_preparation.preprocess import PreProcess
    from seizyml.helper.get_features import compute_features
    from seizyml.train.select_features import select_features
    from tqdm import tqdm
    
    # check if user input exists in process types
    process_type_options = ['compute_features', 'train_model']
    if p is None:
        process_type = set(process_type_options)
    else:
        process_type = set([p])
    
    process_type = list(process_type.intersection(process_type_options))
    if not process_type:
        click.secho(f"\n -> Got'{p}' instead of {process_type_options}\n",
                    fg='yellow', bold=True)
        return
    
    # get train path from settings
    train_path = ctx.obj['parent_path']
    
    # pre-process data and compute features
    if 'compute_features' in process_type:
        
        # get all h5 files with user annotations
        label_files = [x for x in os.listdir(train_path) if x[-4:] == '.csv']
        h5_files = [x.replace('.csv', '.h5') for x in label_files]
        x_all = []
        y_all =[]
        
        # run filecheck
        from seizyml.data_preparation.file_check import train_file_check
        train_file_check(train_path, h5_files, label_files, ctx.obj['win'], ctx.obj['fs'], ctx.obj['channels'])
        print('File check passed.')
        
        # get features
        for x_path, y_path in tqdm(zip(h5_files, label_files), total=len(h5_files)):
            print('-> Cleaning and Computing Features:')
            
            # load f5 file and check if data are properly structured 
            x = load_data(os.path.join(train_path, x_path))
            if x.shape[2] != len(ctx.obj['channels']):
                print('Error! Length of channels:', len(ctx.obj['channels']),
                      'in settings file,', ' does not match train data channels.',
                      x.shape[2], '.')
                return
            if x.shape[1] != int(ctx.obj['fs']*ctx.obj['win']):
                print('Error! fs*win -ie window size-' , int(ctx.obj['fs']*ctx.obj['win']),
                      'in settings file',
                      'does not match train data dimensions.',
                      x.shape[1], '.')
                return
            
            # clean file, compute and normalize features
            obj = PreProcess("", "", fs=ctx.obj['fs'],)
            x_clean = obj.filter_clean(x)
            features_temp, feature_labels = compute_features(x_clean, ctx.obj['features'], ctx.obj['channels'], ctx.obj['fs'])
            features_temp = StandardScaler().fit_transform(features_temp)
            
            # append x and y data
            x_all.append(features_temp)
            y_all.append(np.loadtxt(os.path.join(train_path, y_path)))

        # concantenate and save
        features = np.concatenate(x_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        save_data(os.path.join(train_path, 'features.h5'), features)
        save_data(os.path.join(train_path, 'y.h5'), y)
        np.savetxt(os.path.join(train_path, 'feature_labels.txt'), feature_labels, fmt="%s")
        
    # select features and train model
    if 'train_model' in process_type:
        print('-> Training Model:')
        
        # select features
        if 'features' not in locals():
            feature_labels = np.loadtxt(os.path.join(train_path, 'feature_labels.txt'), dtype=str)
            features = load_data(os.path.join(train_path, 'features.h5'))
            y = load_data(os.path.join(train_path, 'y.h5'))
        
        selected_features = select_features(features, y, feature_labels, r_threshold=ctx.obj['feature_select_thresh'], 
                                        feature_size=ctx.obj['feature_size'], 
                                        nleast_correlated=ctx.obj['nleast_corr'])
        
        # train model
        trained_model_path = os.path.join(train_path, ctx.obj['trained_model_dir'])
        train_df = train_and_save_models(trained_model_path, features, y, selected_features, feature_labels)
        train_df.to_csv(os.path.join(trained_model_path, 'trained_models.csv'), index=False)
        
        # find model with best f1 score and save to settings
        idx = train_df['F1'].idxmax()
        model_id = train_df.loc[idx, 'ID']

        # pass model id and train path to settings
        ctx.obj.update({'model_id': model_id})
        ctx.obj.update({'train_path': ctx.obj['parent_path']})
        print('Best model based on F1 score was selected:', model_id)
    
    # save settings
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj, file)

def cli_entry_point():
    """Entry point for the `seizyml` command when installed via pip."""
    global settings_path
    settings_path = os.path.join('seizyml', 'config.yaml')
    temp_settings_path = os.path.join('seizyml', 'temp_config.yaml')

    if not os.path.isfile(settings_path):
        import shutil
        shutil.copy(temp_settings_path, settings_path)

    else:
        # Check if keys match; if not, reset the settings file
        with open(temp_settings_path, "r") as file:
            temp_settings = yaml.safe_load(file)
        with open(settings_path, "r") as file:
            settings = yaml.safe_load(file)

        if settings.keys() != temp_settings.keys():
            import shutil
            shutil.copy(temp_settings_path, settings_path)
        
    # init cli
    main(obj={})

if __name__ == '__main__':
    cli_entry_point()
