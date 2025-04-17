from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt
import sys

class SeizyMLGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.setWindowTitle("SeizyML - Seizure Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Main Layout
        main_layout = QHBoxLayout(self)

        # Sidebar (Navigation)
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: none;
                font-size: 16px;
                padding: 10px;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 5px;
            }
            QListWidget::item:selected {
                background-color: #00BFFF;
                color: white;
            }
        """)

        # Sidebar Navigation Items
        self.sections = ["Home", "Train Model", "Set Data Path", "Preprocess Data", 
                         "Predict Seizures", "Verify Seizures", "Extract Properties", "Feature Contribution"]
        
        self.sidebar_items = []
        for i, section in enumerate(self.sections):
            item = QListWidgetItem(section)
            self.sidebar.addItem(item)
            self.sidebar_items.append(item)

        # Initially disable all tabs except "Home"
        self.disable_tabs(except_home=True)

        # Stacked Widget (Main Content Area)
        self.pages = QStackedWidget()
        
        # Home Page
        self.home_page = self.create_page("Welcome to SeizyML", "Use the sidebar to navigate.", check_passed=False)
        self.pages.addWidget(self.home_page)

        # Other Pages (Placeholder UI)
        self.train_page = self.create_page("Train Model", "Click 'Train' to start the training process.", check_passed=False)
        self.set_path_page = self.create_page("Set Data Path", "Specify the directory containing EEG data.", check_passed=False)
        self.preprocess_page = self.create_page("Preprocess Data", "Apply filtering and remove artifacts.", check_passed=False)
        self.predict_page = self.create_page("Predict Seizures", "Run seizure detection on preprocessed data.", check_passed=False)
        self.verify_page = self.create_page("Verify Seizures", "Adjust seizure boundaries manually.", check_passed=False)
        self.extract_page = self.create_page("Extract Properties", "Extract seizure characteristics from data.", check_passed=False)
        self.feature_page = self.create_page("Feature Contribution", "Visualize feature importance in seizure classification.", check_passed=False)

        # Add pages to the stacked widget
        self.pages.addWidget(self.train_page)
        self.pages.addWidget(self.set_path_page)
        self.pages.addWidget(self.preprocess_page)
        self.pages.addWidget(self.predict_page)
        self.pages.addWidget(self.verify_page)
        self.pages.addWidget(self.extract_page)
        self.pages.addWidget(self.feature_page)

        # Connect Sidebar Selection to Page Change
        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)

        # Add Sidebar and Pages to Layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages, 1)

    def create_page(self, title, description, check_passed):
        """Creates a placeholder UI for each section."""
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        label = QLabel(title)
        label.setFont(QFont("Arial", 18, QFont.Bold))
        label.setStyleSheet("color: #07a845; padding: 10px;")
        label.setAlignment(Qt.AlignCenter)

        # Description
        desc_label = QLabel(description)
        desc_label.setFont(QFont("Arial", 14))
        desc_label.setAlignment(Qt.AlignCenter)

        # Button (Placeholder for Functionality)
        button = QPushButton(f"Run {title}")
        button.setFont(QFont("Arial", 14))
        button.setStyleSheet("""
            QPushButton {
                background-color: #07a845;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e8c94d;
            }
        """)

        # If it's the home page, clicking the button will enable other tabs
        if title == "Welcome to SeizyML":
            button.setText("Pass Check & Enable Tabs")
            button.clicked.connect(self.enable_tabs)

        layout.addWidget(label)
        layout.addWidget(desc_label)
        layout.addWidget(button, alignment=Qt.AlignCenter)
        page.setLayout(layout)

        return page

    def disable_tabs(self, except_home=False):
        """Disable all tabs except Home if specified, and visually grey them out."""
        for i, item in enumerate(self.sidebar_items):
            if except_home and i == 0:
                continue  # Keep Home enabled
            item.setFlags(Qt.NoItemFlags)  # Completely disable item
            item.setForeground(QColor(100, 100, 100))  # Grey out text

    def enable_tabs(self):
        """Enable all tabs and restore their normal appearance."""
        for item in self.sidebar_items:
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # Restore clickability
            item.setForeground(QColor(255, 255, 255))  # Restore white text

# Run Application
app = QApplication(sys.argv)
window = SeizyMLGUI()
window.show()
sys.exit(app.exec())
