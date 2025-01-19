from PySide6.QtWidgets import (QPushButton, QMenu, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QButtonGroup,\
                               QRadioButton, QProgressBar, QSlider)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
import constants as CONST


class SettingWidget(QWidget):
    def __init__(self, handler, go_result):
        super().__init__()
        self.on_model_change = handler.init_model
        self.set_data = handler.set_data
        self.set_learning_params = handler.set_learning_params
        self.load_model = handler.load_test_model
        self.train_model = handler.train_model
        self.stop_train = handler.stop_train
        self.go_result = go_result

        handler.progress.connect(self.update_progress)
        handler.stopped.connect(self.task_stopped)
        handler.on_test_finish.connect(self.test_finished)

        # Create a button to trigger the dropdown menu
        model_dropdown = self.create_dropdown_menu(CONST.model, [self.on_model_change])
        self.on_model_change(CONST.model[0])

        self.dataset_group = QButtonGroup(self)
        dataset_selection = self.create_radio_group("Dataset", CONST.dataset, self.dataset_group)
        self.dataset_name = CONST.dataset[0]
        self.format_group = QButtonGroup(self)
        input_format = self.create_radio_group("Format", CONST.format, self.format_group)
        self.format = CONST.format[0]

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
        lr_slider.setValue(int((self.lr-self.min_value) / self.precision)+1)
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
        epoch_slider.setMinimum(1)
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

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(lambda: self.on_model_change())
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_test)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.next_btn)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(model_dropdown)
        self.layout.addLayout(dataset_selection)
        self.layout.addLayout(input_format)
        self.layout.addLayout(lr_layout)
        self.layout.addLayout(epoch_layout)
        self.layout.addLayout(train_layout)
        self.layout.addLayout(btn_layout)

    def start_task(self):
        """
        Starts the worker thread to execute the long-running task.
        """
        self.train_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.set_data(self.dataset_name, self.format)
        self.set_learning_params(self.lr, self.epoch)
        self.train_model()

    def stop_task(self):
        """Stop the worker thread."""
        self.stop_train()
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
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

    def test_finished(self):
        """
        Called when the worker thread finishes its task.
        """
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def load_test(self):
        self.set_data(self.dataset_name, self.format)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.load_model()

    def next(self):
        self.load_model()
        self.go_result()

    def task_stopped(self):
        """
        Called when the worker thread finishes its task.
        """
        print("Training stopped.")

    def create_dropdown_menu(self, elements, callbacks, default_idx=0):
        """Create a dropdown menu"""
        layout = QHBoxLayout()

        button = QPushButton(elements[default_idx], self)
        callbacks.append(lambda text: button.setText(text))
        menu = QMenu(self)
        # Add actions to the menu
        for elem in elements:
            self.create_action(menu, elem, callbacks)

        button.setFixedWidth(600)
        menu.setFixedWidth(button.width())
        button.setMenu(menu)  # Set the dropdown menu
        label = QLabel("Model :")
        layout.addWidget(label)
        layout.addWidget(button)

        return layout

    def create_radio_group(self, title, actions, button_group, default_idx=0):
        # Create the radio buttons
        buttons = []
        for action in actions:
            radio_btn = QRadioButton(action, self)
            buttons.append(radio_btn)
            radio_btn.toggled.connect(lambda: self.update_value(title))

        # Add radio buttons to it
        for button in buttons:
            button_group.addButton(button)

        buttons[default_idx].setChecked(True)

        label = QLabel(f'{title} :', self)
        layout = QHBoxLayout()
        layout.addWidget(label)
        for button in buttons:
            layout.addWidget(button)

        return layout

    def update_value(self, category):
        """Update label text based on the selected radio button"""
        if category.lower() == "dataset":
            selected_radio = self.dataset_group.checkedButton()
            self.dataset_name = selected_radio.text()
        elif category.lower() == "format":
            selected_radio = self.format_group.checkedButton()
            self.format = selected_radio.text()

    def create_action(self, menu, text, callbacks):
        action = QAction(text, self)
        action.triggered.connect(lambda: [cb(text) for cb in callbacks])
        menu.addAction(action)
