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
    'default_model': '',
}

def_user_settings = {
    # Model Settings
    'channels': ['eeg1', 'eeg2'],
    'fs': 100,
    'win': 5,
    'trained_model_dir': 'models',
    'features': [
        'line_length', 'kurtosis', 'skewness', 'rms', 'mad', 'var',
        'energy', 'hjorth_mobility', 'hjorth_complexity', 'delta_power',
        'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'weighted_frequency', 'spectral_entropy', 'get_envelope_max_diff'
    ],
    'feature_select_thresh': 0.90,
    'feature_size': [5, 10, 15],

    # App (Data Processing) Settings
    'data_dir': 'data',
    
    'gui_win': 1,
    'nleast_corr': 5,
    'post_processing_method': 'dual_threshold',
    'rolling_window': 6,
    'event_threshold': 0.5,
    'boundary_threshold': 0.2,
    'dilation': 2,
    'erosion': 1,
    'model_predictions_dir': 'model_predictions',
    'data_validated': False,
    'processed_check': False,
    'predicted_check': False,
    'processed_dir': 'processed',
    'verified_predictions_dir': 'verified_predictions',
}

# -------------------- SETTINGS LOADER -------------------- #
def load_yaml_settings(settings_path, default_settings, setting_type="Settings"):
    """
    Load settings from a YAML file or initialize with defaults if the file doesn't exist.
    """
    try:
        if settings_path.is_file():
            with open(settings_path, "r") as file:
                settings = yaml.safe_load(file)

            # Check for key consistency
            missing_keys = set(default_settings) - set(settings)
            if missing_keys:
                click.secho(f"⚠️  Missing keys in {setting_type}: {', '.join(missing_keys)}", fg='yellow')
                for key in missing_keys:
                    settings[key] = default_settings[key]

                # Save updated settings
                with open(settings_path, "w") as file:
                    yaml.dump(settings, file)
                click.secho(f"🛠️  Added missing keys to {setting_type}.", fg='cyan')

            return settings
        else:
            # Create default settings if the file doesn't exist
            with open(settings_path, "w") as file:
                yaml.dump(default_settings, file)
            click.secho(f"✅ Created default {setting_type.lower()} at {settings_path}", fg='green')
            return default_settings

    except yaml.YAMLError as e:
        click.secho(f"❌ YAML error in {settings_path}: {e}", fg='red')
        return default_settings

# -------------------- CORE SETTINGS LOADER -------------------- #
def get_core_settings():
    return load_yaml_settings(core_settings_path, def_core_settings, setting_type="Core Settings")

# -------------------- USER SETTINGS LOADER -------------------- #
def get_user_settings(model_path):
    """
    Load unified user settings (model + app settings) from the model directory.
    """
    return load_yaml_settings(Path(model_path) / user_settings_file, def_user_settings, setting_type="User Settings")

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
