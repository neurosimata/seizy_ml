# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import click
import os
import json
### ----------------------------------------------------------------- ###


def check_main(folder, data_dir, csv_dir):
    """
    Check if folders exist and if h5 files match csv files.

    Parameters
    ----------
    folder : dict, with config settings
    data_dir : str, data directory name
    true_dir : str, csv directory name

    Returns
    -------
    None,str, None if test passes, otherwise a string is returned with the name
    of the folder where the test did not pass

    """
    
    h5_path = os.path.join(folder, data_dir)
    ver_path = os.path.join(folder, csv_dir)
    if not os.path.exists(h5_path):
        return h5_path
    if not os.path.exists(ver_path):
        return ver_path
    h5 = {x.replace('.h5', '') for x in os.listdir(h5_path)}
    ver = {x.replace('.csv', '') for x in os.listdir(ver_path)}   
    if len(h5) != len(h5 & ver):
        return folder

@click.group()
@click.pass_context
def main(ctx):
    """
    -----------------------------------------------------
    
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

    ----------------------------------------------------- 
                                                                                                                                           
    """
        
    # get settings and pass to context
    with open(settings_path, 'r') as file:
        settings = json.loads(file.read())
        ctx.obj = settings.copy()
    
@main.command()
@click.pass_context
def setpath(ctx):
    """1: Set path"""
    
    path = input('Enter Parent path: \n')
    ctx.obj.update({'parent_path': path})
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj))  
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg='green', bold=True)    
        
@main.command()
@click.pass_context
def filecheck(ctx):
    """2: Check whether files can be opened and read"""
    
    # get child folders and create success list for each folder
    if not os.path.exists(ctx.obj['parent_path']):
        click.secho(f"\n -> Parent path '{ctx.obj['parent_path']}' was not found." +\
                    " Please run -setpath-.\n",
                    fg='yellow', bold=True)
        return
    
    ### code to check for files ###

    # save error check to settings file
    ctx.obj.update({'file_check': True})
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj)) 
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
    ctx.obj.update({'processed':True})

    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj)) 
    return
 
@main.command()
@click.pass_context
def predict(ctx):
    """4: Generate model predictions"""
    
    if ctx.obj['processed'] == False:
        click.secho("\n -> Data need to be preprocessed first. Please run -preprocess-.\n",
                    fg='yellow', bold=True)
        return
    
    from data_preparation.get_predictions import ModelPredict
    # get paths and model predictions
    load_path = os.path.join(ctx.obj['parent_path'], ctx.obj['processed_dir'])
    save_path = os.path.join(ctx.obj['parent_path'], ctx.obj['model_predictions_dir'])
    model_obj = ModelPredict(load_path, save_path, win=ctx.obj['win'], fs=ctx.obj['fs'])
    model_obj.predict()
    ctx.obj.update({'predicted':1})
    
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj)) 
    return

@main.command()
@click.pass_context
def verify(ctx):
    """5: Verify detected seizures"""
    
    if ctx.obj['predicted'] == False:
        click.secho("\n -> Model predictions have not been generated. Please run -predict-.\n",
                    fg='yellow', bold=True)
        return

    out = check_main(folder=ctx.obj['parent_path'],
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
    temp_settings_path = 'temp_config.json'
    settings_path = 'config.json'
    
    # check if settings file exist and if all the fields are present
    if not os.path.isfile(settings_path):
        import shutil
        shutil.copy(temp_settings_path, settings_path)
        
    else:
        # check if keys match otherwise load original settings
        with open(temp_settings_path, 'r') as file:
            temp_settings = json.loads(file.read())      
        with open(settings_path, 'r') as file:
            settings = json.loads(file.read())
    
        if settings.keys() != temp_settings.keys():
            import shutil
            shutil.copy(temp_settings_path, settings_path)
        
    # init cli
    main(obj={})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    