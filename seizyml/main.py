from PySide6.QtWidgets import QApplication, QFileDialog
import sys
import yaml
from app import SeizyMLGUI
import backend  # Only imported in main.py
from default_settings import def_user_settings

def load_settings():
    """Loads settings from a YAML file and returns a dictionary."""
    with open(r"D:\test_seizyml\train_data\user_settings.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def connect_functions(window):
    """Connects backend functions to the GUI components."""
    train_tab = window.train_page  # Get reference to the TrainTab instance

    # Handle Browse button click (now separate from GUI)
    def handle_directory_selection():
        """Handles directory selection and validation."""
        dir_path = QFileDialog.getExistingDirectory(train_tab, "Select Data Directory")
        if dir_path:
            train_tab.path_input.setText(dir_path)
            is_valid = backend.validate_directory(dir_path)  # Call backend function
            train_tab.path_set_signal.emit(is_valid)  # Send result back to GUI

    train_tab.browse_button.clicked.connect(handle_directory_selection)
    train_tab.train_button.clicked.connect(lambda: backend.start_training(train_tab))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # load settings from yaml or use defaults
    settings = def_user_settings
    # settings = load_settings()  # Load settings from YAML
    window = SeizyMLGUI(settings)  # Pass settings to GUI

    connect_functions(window)  # Connect backend logic

    window.show()
    sys.exit(app.exec())
