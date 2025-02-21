import os

def validate_directory(path):
    """Checks if the provided path is a valid directory."""
    return os.path.exists(path) and os.path.isdir(path)

def start_training(train_tab):
    """Simulated training function that updates UI."""
    train_tab.notification_field.append("🚀 Training started...")
    # Placeholder for actual training logic.
