from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QStackedWidget, QLabel
import sys
from GUI import *
from model_handler import ModelHandler


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.handler = ModelHandler()
        self.handler.result_ready.connect(self.go_results)

        self.setWindowTitle("Human Action Recognition")

        # Create the stacked widget to hold different layouts
        self.stacked_widget = QStackedWidget()

        # First layout configuration
        self.setting_widget = SettingWidget(self.handler)

        # Second layout configuration
        self.res_widget = ResWidget(self.handler.predict)

        # Add both layouts to the stacked widget
        self.stacked_widget.addWidget(self.setting_widget)
        self.stacked_widget.addWidget(self.res_widget)

        # Set the initial layout (layout 1 will be shown first)
        self.stacked_widget.setCurrentIndex(0)

        # Set the stacked widget as the central widget
        self.setCentralWidget(self.stacked_widget)


    def go_results(self):
        self.res_widget.setPic()
        self.stacked_widget.setCurrentIndex(1)


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 600)  # Set the window size
    window.show()
    sys.exit(app.exec())
