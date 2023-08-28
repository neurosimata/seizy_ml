# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import click
import os
import yaml
### ----------------------------------------------------------------- ###

class CustomOrderGroup(click.Group):
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
    \b             _ __  ___ _ _____   _| .  . || |    
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
    from data_preparation.file_check import check_main
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
def train(ctx, p):
    """* Train Model """
    
    # check if user input exists in process types
    process_type_options = ['preprocess', 'get_features', 'parameter_space', 'train']
    if p is None:
        process_type = set(process_type_options)
    else:
        process_type = set([p])
    
    process_type = list(process_type.intersection(process_type_options))
    if not process_type:
        click.secho(f"\n -> Got'{p}' instead of {process_type_options}\n",
                    fg='yellow', bold=True)
        return
        
    train_path = os.path.join(ctx.obj['parent_path'], ctx.obj['train_dir'])
    if 'preprocess' in process_type:
        from helper.io import load_data, save_data
        from helper.preprocess import PreProcess
        # load data and preprocess
        data = load_data(os.path.join(train_path, 'x_pre.h5'))
        obj = PreProcess("", "", fs=ctx.obj['fs'],)
        x = obj.filter_clean(data)
        save_data(os.path.join(train_path, 'x.h5'), x)

    if 'get_features' in process_type:
        from helper.get_features import compute_features
        features_array, feature_labels = compute_features(data, ctx.obj['single_channel_features'], ctx.obj['cross_channel_features'],
                         ctx.obj['channels'])
        
    if 'parameter_space' in process_type:
        from get_train_parameters import feature_selection_and_ranking, feature_space
        # get feature metrics
        y_true = load_data(os.path.join(train_path, 'x_pre.h5'))
        feature_metrics = feature_selection_and_ranking(x, y_true, feature_labels)
        feature_space, ranks = feature_space(feature_metrics, feature_size=[4,8],
                          export_name='feature_space.csv', export=True)
        
    if 'train' in process_type:
        print('train')

        
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
    from data_preparation.file_check import check_h5_files
    error = check_h5_files(os.path.join(ctx.obj['parent_path'], ctx.obj['data_dir']),
                           win=ctx.obj['win'], fs=ctx.obj['fs'])
    
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
    
    from data_preparation.preprocess import PreProcess
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
    
    from data_preparation.get_predictions import ModelPredict
    # get paths and model predictions
    load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
    save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['model_predictions_dir'])
    model_obj = ModelPredict(load_path, save_path, win=ctx.obj['win'], fs=ctx.obj['fs'])
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
    
    from data_preparation.file_check import check_verified
    out = check_verified(folder=ctx.obj['parent_path'],
                     data_dir=ctx.obj['processed_dir'],
                     csv_dir=ctx.obj['model_predictions_dir'])
    if out:
        click.secho(f"\n -> Error. Could not find: {out}.\n",
             fg='yellow', bold=True)
        return
    
    # Create instance for UserVerify class
    from user_gui.user_verify import UserVerify
    obj = UserVerify(ctx.obj['parent_path'],
                     ctx.obj['processed_dir'], 
                     ctx.obj['model_predictions_dir'],
                     ctx.obj['verified_predictions_dir'])
    file_id = obj.select_file()                     # user file selection
    data, idx_bounds = obj.get_bounds(file_id)      # get data and seizure index
    
    # check for zero seizures otherwise proceed with gui creation
    if idx_bounds.shape[0] == 0:
        obj.save_emptyidx(data.shape[0], file_id)     
    else:
        from user_gui.verify_gui import VerifyGui
        VerifyGui(ctx.obj, file_id, data, idx_bounds)
        
        
@main.command()
@click.pass_context
def extractproperties(ctx):
    """6: Get seizure properties"""
    
    ver_path = os.path.join(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'])
    if  os.path.exists(ver_path):
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))

    if not filelist:
        click.secho("\n -> Could not find verified seizures: Please verify detected seizures.\n",
             fg='yellow', bold=True)
        return
    
    # get properies and save
    from helper.get_seizure_properties import get_seizure_prop
    _, save_path = get_seizure_prop(ctx.obj['parent_path'], ctx.obj['verified_predictions_dir'], ctx.obj['win'])
    click.secho(f"\n -> Properies were saved in '{save_path}'.\n", fg='green', bold=True)
    
    
if __name__ == '__main__':
    
    # define settings path
    temp_settings_path = 'temp_config.yaml'
    settings_path = 'config.yaml'
    
    # check if settings file exist and if all the fields are present
    if not os.path.isfile(settings_path):
        import shutil
        shutil.copy(temp_settings_path, settings_path)
        
    else:
        # check if keys match otherwise load original settings
        with open(temp_settings_path, 'r') as file:
            temp_settings = yaml.safe_load(file)      
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file) 
    
        if settings.keys() != temp_settings.keys():
            import shutil
            shutil.copy(temp_settings_path, settings_path)
        
    # init cli
    main(obj={})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    