from pathlib import Path
app_directory = Path(__file__).resolve().parent
core_settings_path = app_directory / 'core_settings.yaml'
settings_file_name = 'settings.yaml'
default_settings = {
    'channels': ['eeg1', 'eeg2'],
    'data_dir': 'data',
    'file_check': False,
    'fs': 100,
    'features': [
        'line_length',
        'kurtosis',
        'skewness',
        'rms',
        'mad',
        'var',
        'energy',
        'hjorth_mobility',
        'hjorth_complexity',
        'delta_power',
        'theta_power',
        'alpha_power',
        'beta_power',
        'gamma_power',
        'weighted_frequency',
        'spectral_entropy',
        'get_envelope_max_diff'
    ],
    'feature_select_thresh': 0.90,
    'feature_size': [5, 10, 15],
    'gui_win': 1,
    'nleast_corr': 5,
    'post_processing_method': 'dual_threshold',
    'rolling_window': 6,
    'event_threshold': 0.5,
    'boundary_threshold': 0.2,
    'dilation': 2,
    'erosion': 1,
    'model_path': '',
    'model_predictions_dir': 'model_predictions',
    'settings_path': '',
    'parent_path': '',
    'predicted_check': False,
    'processed_check': False,
    'trained_model_dir': 'models',
    'processed_dir': 'processed',
    'verified_predictions_dir': 'verified_predictions',
    'win': 5
}