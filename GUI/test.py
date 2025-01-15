from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QProgressBar
from PySide6.QtCore import QTimer
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading Bar Example")

        # Create a button to simulate a task
        self.start_button = QPushButton("Start Task", self)
        self.start_button.clicked.connect(self.start_task)  # Connect button to task start

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)

        # Create a central widget to hold the layout
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Timer for simulating progress
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        # Track progress
        self.progress = 0

    def start_task(self):
        """Start the task and begin updating the progress bar."""
        self.progress = 0
        self.progress_bar.setValue(self.progress)
        self.timer.start(100)  # Update every 100 milliseconds

    def update_progress(self):
        """Update the progress bar with the current progress."""
        self.progress += 1
        self.progress_bar.setValue(self.progress)

        # If progress reaches 100, stop the timer
        if self.progress >= 100:
            self.timer.stop()
            self.start_button.setText("Task Completed")

if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(300, 150)  # Set window size
    window.show()
    sys.exit(app.exec())
