from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QStackedWidget, QLabel
import sys
from GUI import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Action Recognition")

        # Create the stacked widget to hold different layouts
        self.stacked_widget = QStackedWidget()

        # First layout configuration
        self.start_widget = StartWidget()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(lambda: self.load_model(True))
        train_btn = QPushButton("Train")
        train_btn.clicked.connect(self.go_train_settings)
        self.start_widget.add_button(load_btn)
        self.start_widget.add_button(train_btn)

        # Third layout configuration
        self.res_widget = ResWidget()
        ResWidget().init_model(self.start_widget.get_model_name())

        self.setting_widget = SettingWidget(self.load_model)

        # Add both layouts to the stacked widget
        self.stacked_widget.addWidget(self.start_widget)
        self.stacked_widget.addWidget(self.setting_widget)
        self.stacked_widget.addWidget(self.res_widget)

        # Set the initial layout (layout 1 will be shown first)
        self.stacked_widget.setCurrentIndex(0)

        # Set the stacked widget as the central widget
        self.setCentralWidget(self.stacked_widget)

    def go_train_settings(self):
        """Switch to layout 1"""
        self.stacked_widget.setCurrentIndex(1)

    def load_model(self, goto_results=False):
        ResWidget().load_model(StartWidget.get_model_name())
        ResWidget().setPic()
        if goto_results:
            self.stacked_widget.setCurrentIndex(2)

if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 600)  # Set the window size
    window.show()
    sys.exit(app.exec())
