from PySide6.QtWidgets import QPushButton, QMenu, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QButtonGroup, QRadioButton, \
    QProgressBar, QSlider, QSizePolicy
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
import constants as CONST
from GUI import *
from GUI.TrainerThread import TrainerThread


class SettingWidget(QWidget):
    _instance = None  # Class variable to hold the single instance

    def __new__(self, *args, **kwargs):
        if not self._instance:  # Check if an instance already exists
            self._instance = super(SettingWidget, self).__new__(self)  # Create the instance
        return self._instance  # Return the existing instance

    def __init__(self, load_model):
        if not hasattr(self, '_initialized'):
            super().__init__()

            self.load_model = load_model

            self.dataset_group = QButtonGroup(self)
            dataset_selection = self.create_radio_group("Dataset", CONST.dataset, self.dataset_group)
            self.format_group = QButtonGroup(self)
            input_format = self.create_radio_group("Format", CONST.format, self.format_group)

            # Define the range and precision
            self.min_value = 0.0001
            self.max_value = 0.01
            self.precision = 0.0001

            # Calculate the slider range
            self.slider_steps = int((self.max_value - self.min_value) / self.precision)

            # Create the slider
            self.lr = 0.0006
            lr_slider = QSlider(Qt.Horizontal)
            lr_slider.setMaximumHeight(20)
            lr_slider.setMinimum(0)
            lr_slider.setMaximum(self.slider_steps)
            lr_slider.setValue(int((self.lr-self.min_value) / self.precision))
            lr_slider.valueChanged.connect(self.update_lr)

            # Create a label to display the current value
            self.lr_label = QLabel(f"lr : {self.lr:.4f}")
            self.lr_label.setMaximumHeight(20)

            lr_layout = QHBoxLayout()
            lr_slider.setMaximumHeight(20)
            lr_layout.addWidget(lr_slider)
            lr_layout.addWidget(self.lr_label)

            # Create the slider
            self.epoch = 30
            epoch_slider = QSlider(Qt.Horizontal)
            epoch_slider.setMaximumHeight(20)
            epoch_slider.setMinimum(0)
            epoch_slider.setMaximum(100)
            epoch_slider.setValue(self.epoch)
            epoch_slider.valueChanged.connect(self.update_epoch)

            # Create a label to display the current value
            self.epoch_label = QLabel(f"epochs : {self.epoch}")
            self.epoch_label.setMaximumHeight(20)

            epoch_layout = QHBoxLayout()
            epoch_slider.setMaximumHeight(20)
            epoch_layout.addWidget(epoch_slider)
            epoch_layout.addWidget(self.epoch_label)

            # Create the progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)      # Set range from 0 to 100
            self.progress_bar.setValue(0)                           # Initial value is 0
            self.progress_bar.setTextVisible(True)                  # Show the percentage in the bar


            train_layout = QHBoxLayout()
            train_layout.addWidget(self.progress_bar)
            self.train_btn = QPushButton("Train")
            self.train_btn.clicked.connect(self.start_task)
            train_layout.addWidget(self.train_btn)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self.stop_task)
            train_layout.addWidget(self.stop_btn)

            self.load_btn = QPushButton("Load")
            self.load_btn.clicked.connect(self.skip_train)
            self.next_btn = QPushButton("Next")
            self.next_btn.clicked.connect(self.next)

            btn_layout = QHBoxLayout()
            btn_layout.addWidget(self.load_btn)
            btn_layout.addWidget(self.next_btn)

            self.layout = QVBoxLayout(self)
            self.layout.addLayout(dataset_selection)
            self.layout.addLayout(input_format)
            self.layout.addLayout(lr_layout)
            self.layout.addLayout(epoch_layout)
            self.layout.addLayout(train_layout)
            self.layout.addLayout(btn_layout)

            self.trainer_thread = None
            self._initialized = True  # Mark the object as initialized


    def start_task(self):
        """
        Starts the worker thread to execute the long-running task.
        """
        self.train_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        print(f"epochs : {self.epoch} - lr : {self.lr} - format : {self.format} - Dataset : {self.dataset}")
        self.trainer_thread = TrainerThread(ResWidget().model, self.dataset, self.format, self.lr, self.epoch)
        self.trainer_thread.progress.connect(self.update_progress)
        self.trainer_thread.stopped.connect(self.task_stopped)
        self.trainer_thread.finished.connect(self.task_finished)
        self.trainer_thread.start()

    def stop_task(self):
        """Stop the worker thread."""
        self.trainer_thread.stop()
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

    def update_progress(self, value):
        """
        Updates the progress label based on the signal from the worker thread.
        """
        self.progress_bar.setValue(value)

    def update_lr(self, value):
        """Update the label with the current floating-point value."""
        float_value = self.min_value + value * self.precision
        self.lr_label.setText(f"lr : {float_value:.4f}")
        self.lr = float_value

    def update_epoch(self, value):
        self.epoch_label.setText(f"epochs : {value}")
        self.epoch = value

    def task_finished(self):
        """
        Called when the worker thread finishes its task.
        """
        if self.trainer_thread.is_running():
            self.stop_btn.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.load_model(True)

    def skip_train(self):
        """
        Called when the worker thread finishes its task.
        """
        self.load_model()

    def next(self):
        """
        Called when the worker thread finishes its task.
        """
        self.load_model(True)

    def task_stopped(self):
        """
        Called when the worker thread finishes its task.
        """
        print("Training stopped.")

    def create_dropdown_menu(self, title, actions):
        menu = QMenu(self)

        layout = QHBoxLayout()
        label = QLabel(f'{title} :', self)
        layout.addWidget(label)
        button = QPushButton(actions[0], self)

        for action in actions:
            menu.addAction(self.create_action(button, action))

        button.setMenu(menu)
        layout.addWidget(button)

        return layout

    def create_radio_group(self, title, actions, button_group):
        # Create the radio buttons
        buttons = []
        for action in actions:
            radio_btn = QRadioButton(action, self)
            buttons.append(radio_btn)
            radio_btn.toggled.connect(lambda : self.update_value(title))

        # Add radio buttons to it
        for button in buttons:
            button_group.addButton(button)

        buttons[0].setChecked(True)

        label = QLabel(f'{title} :', self)
        layout = QHBoxLayout()
        layout.addWidget(label)
        for button in buttons:
            layout.addWidget(button)

        return layout

    def update_value(self, category):
        """Update label text based on the selected radio button"""
        if(category.lower()=="dataset"):
            selected_radio = self.dataset_group.checkedButton()
            self.dataset = selected_radio.text()
            print(f"Dataset value : {self.dataset}")
        elif(category.lower()=="format"):
            selected_radio = self.format_group.checkedButton()
            self.format = selected_radio.text()
            print(f"Format value : {self.format}")

    def create_action(self, button, text):
        """Create an action for the dropdown menu"""
        action = QAction(text, self)
        action.triggered.connect(lambda: button.setText(text))  # Connect to the handler
        return action

    def add_widget(self, widget):
        return self.layout.addWidget(widget)

    def add_layout(self, layout):
        return self.layout.addLayout(layout)
