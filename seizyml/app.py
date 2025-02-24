import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QListWidget, QListWidgetItem, QLineEdit, QTextEdit, QSplashScreen
)
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtCore import Qt, Signal, QTimer

class SeizyMLGUI(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("SeizyML - Seizure Detection")
        self.setGeometry(100, 100, 800, 600)
        qr=self.frameGeometry()           
        cp=QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        # Load external style sheet
        with open("style.qss", "r") as f:
            self.setStyleSheet(f.read())

        main_layout = QHBoxLayout(self)

        # Sidebar (Navigation)
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setCursor(QCursor(Qt.PointingHandCursor))
        self.sections = ["Home", "Train Model", "Load Data", "Preprocess Data"]
        for section in self.sections:
            item = QListWidgetItem(section)
            self.sidebar.addItem(item)

        # Stacked Widget (Pages)
        self.pages = QStackedWidget()
        self.home_page = HomeTab()
        self.pages.addWidget(self.home_page)
        self.train_page = TrainTab(settings)
        self.pages.addWidget(self.train_page)
        self.load_model_page = LoadTab(settings)
        self.pages.addWidget(self.load_model_page)
        self.preprocess_page = PreProcessTab(settings)
        self.pages.addWidget(self.preprocess_page)

        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages, 1)

class HomeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Logo wrapper layout to center horizontally
        logo_layout = QHBoxLayout()
        self.logo_label = QLabel(self)
        pixmap = QPixmap("images/seizyML_logo.png")
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setScaledContents(True)
        self.logo_label.setFixedSize(500, 250)
        self.logo_label.setAlignment(Qt.AlignCenter)

        # Add logo to horizontal layout for centering
        logo_layout.addStretch()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch()

        # Title
        self.title_label = QLabel("Welcome")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Description
        self.desc_label = QLabel("Use the sidebar to navigate.")
        self.desc_label.setObjectName("descLabel")
        self.desc_label.setAlignment(Qt.AlignCenter)

        # Button
        self.get_started_button = QPushButton("Get Started")
        self.get_started_button.setCursor(QCursor(Qt.PointingHandCursor))

        # Add widgets to main layout
        layout.addLayout(logo_layout)  # Ensures logo is centered
        layout.addWidget(self.title_label)
        layout.addWidget(self.desc_label)
        layout.addWidget(self.get_started_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

class TrainTab(QWidget):
    path_set_signal = Signal(bool)  # Emits True/False when path is validated

    def __init__(self, settings):
        super().__init__()
        layout = QVBoxLayout()
        self.title_label = QLabel("Train Model")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Directory Selection
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select training directory...")
        self.path_input.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(QCursor(Qt.PointingHandCursor))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)

        # Settings Fields
        settings_layout = QVBoxLayout()
        self.channels_label = QLabel("Channels:")
        self.channels_input = QLineEdit()
        settings_layout.addWidget(self.channels_label)
        settings_layout.addWidget(self.channels_input)

        self.sampling_rate_label = QLabel("Sampling Rate (Hz):")
        self.sampling_rate_input = QLineEdit()
        settings_layout.addWidget(self.sampling_rate_label)
        settings_layout.addWidget(self.sampling_rate_input)

        self.window_size_label = QLabel("Window Size (sec):")
        self.window_size_input = QLineEdit()
        settings_layout.addWidget(self.window_size_label)
        settings_layout.addWidget(self.window_size_input)

        # Populate fields with settings
        self.populate_fields(settings)

        # Notification Field
        self.notification_field = QTextEdit()
        self.notification_field.setReadOnly(True)
        self.notification_field.setPlaceholderText("Logs and progress updates will appear here...")
        self.notification_field.setMaximumHeight(150)

        # Train Button (Initially Disabled)
        self.train_button = QPushButton("Train Model")
        self.train_button.setEnabled(False)
        self.train_button.setCursor(QCursor(Qt.PointingHandCursor))

        layout.addWidget(self.title_label)
        layout.addLayout(path_layout)
        layout.addLayout(settings_layout)
        layout.addWidget(self.notification_field)
        layout.addWidget(self.train_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        # Connect signals
        self.browse_button.clicked.connect(self.emit_load_directory_signal)
        self.path_set_signal.connect(self.update_train_button)

    def populate_fields(self, settings):
        self.channels_input.setText(", ".join(settings.get("channels", [])))
        self.sampling_rate_input.setText(str(settings.get("sampling_rate", "")))
        self.window_size_input.setText(str(settings.get("window_size", "")))

    def emit_load_directory_signal(self):
        self.path_set_signal.emit(False)

    def update_train_button(self, is_valid):
        self.train_button.setEnabled(is_valid)
        if is_valid:
            self.notification_field.append(f"✅ Directory set: {self.path_input.text()}")
        elif self.path_input.text():
            self.notification_field.append("❌ Invalid directory. Please select a valid data folder.")

class LoadTab(QWidget):
    path_set_signal = Signal(bool)

    def __init__(self, settings):
        super().__init__()
        layout = QVBoxLayout()
        self.title_label = QLabel("Load Model")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Model Selection
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Load Model...")
        self.model_path.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(QCursor(Qt.PointingHandCursor))
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.browse_button)

        # Data selection
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select data directory...")
        self.path_input.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(QCursor(Qt.PointingHandCursor))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)

        # Load Button (Initially Disabled)
        self.load_button = QPushButton("Load")
        # self.load_button.setEnabled(False)
        self.load_button.setCursor(QCursor(Qt.PointingHandCursor))

        # Notification Field
        self.notification_field = QTextEdit()
        self.notification_field.setReadOnly(True)
        self.notification_field.setPlaceholderText("Logs and progress updates will appear here...")
        self.notification_field.setMaximumHeight(150)

        layout.addWidget(self.title_label)
        layout.addLayout(model_layout)
        layout.addLayout(path_layout)
        layout.addWidget(self.notification_field)
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)

class PreProcessTab(QWidget):
    path_set_signal = Signal(bool)

    def __init__(self, settings):
        super().__init__()
        layout = QVBoxLayout()
        self.title_label = QLabel("Preprocess Data")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Directory Selection
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select data directory...")
        self.path_input.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(QCursor(Qt.PointingHandCursor))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)

        # Notification Field
        self.notification_field = QTextEdit()
        self.notification_field.setReadOnly(True)
        self.notification_field.setPlaceholderText("Logs and progress updates will appear here...")
        self.notification_field.setMaximumHeight(150)

        # PreProcess Button (Initially Disabled)
        self.preprocess_button = QPushButton("Preprocess")
        self.preprocess_button.setEnabled(False)
        self.preprocess_button.setCursor(QCursor(Qt.PointingHandCursor))

        layout.addWidget(self.title_label)
        layout.addLayout(path_layout)
        layout.addWidget(self.notification_field)
        layout.addWidget(self.preprocess_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)

if __name__ == "__main__":
    settings = {
        "channels": ["Fp1", "Fp2", "F3", "F4"],
        "sampling_rate": 256,
        "window_size": 2
    }

    # Create and show splash screen
    app = QApplication(sys.argv)
    splash_pix = QPixmap("images/seizyML_logo.png").scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    window = SeizyMLGUI(settings)
    QTimer.singleShot(500, splash.close)
    QTimer.singleShot(500, window.show)
    sys.exit(app.exec())