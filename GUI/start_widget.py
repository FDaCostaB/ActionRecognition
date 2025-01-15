from PySide6.QtWidgets import QPushButton, QMenu, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QProgressBar
from PySide6.QtGui import QAction
import constants as CONST
from GUI import ResWidget


class StartWidget(QWidget):
    _instance = None  # Class variable to hold the single instance

    def __new__(self, *args, **kwargs):
        if not self._instance:  # Check if an instance already exists
            self._instance = super(StartWidget, self).__new__(self)  # Create the instance
        return self._instance  # Return the existing instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()

            self.model_name = None
            # Create a button to trigger the dropdown menu
            model_layout = self.create_dropdown_menu(CONST.model)
            self.btn_layout = QHBoxLayout()

            # Set up the layout and central widget
            self.layout = QVBoxLayout(self)
            self.layout.addLayout(model_layout)
            self.layout.addLayout(self.btn_layout)

            self.setLayout(self.layout)

            self._initialized = True  # Mark the object as initialized

    def create_dropdown_menu(self, actions):
        """Create a dropdown menu"""
        self.model_name = actions[0]
        layout = QHBoxLayout()

        button = QPushButton(actions[0], self)
        menu = QMenu(self)
        # Add actions to the menu
        for action in actions:
            menu.addAction(StartWidget.create_action(self, button, action))

        button.setFixedWidth(600)
        menu.setFixedWidth(button.width())
        button.setMenu(menu)  # Set the dropdown menu
        label = QLabel("Model :")
        layout.addWidget(label)
        layout.addWidget(button)

        return layout

    @staticmethod
    def create_action(parent, button, text):
        action = QAction(text, parent)
        action.triggered.connect(lambda: StartWidget.select_model(button, text))  # Connect to the handler
        return action

    @staticmethod
    def select_model(button, text):
        button.setText(text)
        StartWidget().model_name = text

    @staticmethod
    def add_widget(widget):
        StartWidget().layout.addWidget(widget)

    @staticmethod
    def add_button(widget):
        StartWidget().btn_layout.addWidget(widget)

    @staticmethod
    def add_layout(layout):
        StartWidget().layout.addWidget(layout)

    @staticmethod
    def get_model_name():
        return StartWidget().model_name
