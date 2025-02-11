#### --------------- Imports --------------- ####
import click
from pathlib import Path
import yaml
#### --------------------------------------- ####

# -------------------- FILE PATHS -------------------- #
core_settings_file = 'core_settings.yaml'
user_settings_file = 'user_settings.yaml'
app_directory = Path(__file__).resolve().parent
core_settings_path = app_directory / core_settings_file

# -------------------- DEFAULT SETTINGS -------------------- #
def_core_settings = {
    'parent_path': '',
    'user_settings_path': '',
    'model_path': '',
    # app internal checks
    'data_validated': False,
    'is_processed': False,
    'is_predicted': False,
}

def_user_settings = {
    # data settings
    'channels': ['eeg1', 'eeg2'],
    'fs': 100,
    'win': 5,

    # gui settings
    'gui_win': 1,
    
    # features
    'features': [
        'line_length', 'kurtosis', 'skewness', 'rms', 'mad', 'var',
        'energy', 'hjorth_mobility', 'hjorth_complexity', 'delta_power',
        'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'weighted_frequency', 'spectral_entropy', 'get_envelope_max_diff'
    ],
    'feature_select_thresh': 0.90,
    'feature_size': [5, 10, 15],
    'nleast_corr': 5,

    # post processing settings
    'post_processing_method': 'dual_threshold',
    'rolling_window': 6,
    'event_threshold': 0.5,
    'boundary_threshold': 0.2,
    'dilation': 2,
    'erosion': 1,

    # app directory names
    'data_dir': 'data',
    'processed_dir': 'processed',
    'model_predictions_dir': 'model_predictions',
    'verified_predictions_dir': 'verified_predictions',
    'trained_model_dir': 'models',
}

# -------------------- SETTINGS LOADER -------------------- #
def get_core_settings():
    """
    Load settings from a YAML file or initialize with defaults if the file doesn't exist.
    """
    try:
        if core_settings_path.is_file():
            with open(core_settings_path, "r") as file:
                settings = yaml.safe_load(file)
            return settings
        else:
            with open(core_settings_path, "w") as file:
                yaml.dump(def_core_settings, file)
            click.secho(f"✅ Created default 'core settings' at {core_settings_path}", fg='green')
            return def_core_settings
        
    except yaml.YAMLError as e:
        click.secho(f"❌ YAML error in {core_settings_path}: {e}", fg='red')
        return def_core_settings

def get_user_settings(settings_path):
    """
    Load settings from a YAML file or initialize with defaults if the file doesn't exist.
    """
    try:
        if settings_path.is_file():
            with open(settings_path, "r") as file:
                settings = yaml.safe_load(file)

            # Check for key consistency
            missing_keys = set(def_user_settings) - set(settings)
            if missing_keys:
                click.secho(f"⚠️  Missing keys in 'user_settings': {', '.join(missing_keys)}", fg='yellow')

            return settings
        else:
            click.secho(f"✅ Getting default 'user_settings'", fg='green')
            return def_user_settings

    except yaml.YAMLError as e:
        click.secho(f"❌ YAML error in {settings_path}: {e}", fg='red')
        return def_user_settings

# -------------------- SAVE SETTINGS FUNCTION -------------------- #
def save_settings(settings_path, settings):
    """
    Save settings to the specified YAML file.

    Parameters:
    - settings_path (str or Path): Path to the YAML file where settings will be saved.
    - settings (dict): The settings dictionary to be saved.

    Returns:
    - None
    """
    settings_path = Path(settings_path)
    with open(settings_path, 'w') as file:
        yaml.dump(settings, file)

# -------------------- CUSTOM CLICK GROUP -------------------- #
class CustomOrderGroup(click.Group):
    """
    A custom Click group that maintains the order of commands added to it.
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
