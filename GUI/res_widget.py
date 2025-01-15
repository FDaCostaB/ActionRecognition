import os
import cv2
import numpy as np
import constants as CONST
import files_utils as futils
from data_loader import Dataset
from models import Model
from lr_scheduler import lrScheduler
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Slot
import model_trainer

class ResWidget(QtWidgets.QWidget):
    _instance = None  # Class variable to hold the single instance

    def __new__(self, *args, **kwargs):
        if not self._instance:  # Check if an instance already exists
            self._instance = super(ResWidget, self).__new__(self)  # Create the instance
        return self._instance  # Return the existing instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()

            self.model = None

            self.folder_path = "datasets/Stanford40/JPEGImages/"
            # List all files in the folder
            self.files, _ = futils.parse_filelist_Stanford40("./datasets/Stanford40/test.txt", CONST.keep_stanford40)
            self.index = 0

            self.pred_1 = self.pred_label()
            self.pred_2 = self.pred_label()
            self.pred_3 = self.pred_label()

            self.pic = QtWidgets.QLabel()
            self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pixmap = None

            self.next_btn = QtWidgets.QPushButton("Next")
            self.prev_btn = QtWidgets.QPushButton("Previous")

            self.next_btn.clicked.connect(self.next)
            self.prev_btn.clicked.connect(self.prev)

            self.btn_layout = QtWidgets.QHBoxLayout()
            self.btn_layout.addWidget(self.prev_btn)
            self.btn_layout.addWidget(self.next_btn)

            self.label_layout = QtWidgets.QVBoxLayout()
            self.label_layout.addWidget(self.pred_1)
            self.label_layout.addWidget(self.pred_2)
            self.label_layout.addWidget(self.pred_3)

            self.layout = QtWidgets.QVBoxLayout(self)
            self.layout.addLayout(self.label_layout)
            self.layout.addWidget(self.pic)
            self.layout.addLayout(self.btn_layout)

            self.batch_amount = None
            self._initialized = True  # Mark the object as initialized

    @Slot()
    def next(self):
        self.index = (self.index + 1) % len(self.files)
        self.setPic()
        return

    @Slot()
    def prev(self):
        self.index = (self.index - 1) % len(self.files)
        self.setPic()
        return

    def setPic(self):
        self.pixmap = QtGui.QPixmap(os.path.join(self.folder_path, self.files[self.index]))
        self.pixmap = self.pixmap.scaled(300, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        category = ('_'.join(self.files[self.index].split('_')[:-1]))
        res = self.predict(self.model.model, str(os.path.join(self.folder_path, self.files[self.index])))

        self.display_prediction(self.pred_1, res[2], category)
        self.display_prediction(self.pred_2, res[1], category)
        self.display_prediction(self.pred_3, res[0], category)

        self.pic.setPixmap(self.pixmap)
        return

    @staticmethod
    def init_model(model_name):
        print(f"Model initialized : {model_name}")
        ResWidget().model = Model(model_name, (112, 112, 3), None, 12)

    @staticmethod
    def load_model(model_name):
        print(f"Model loaded : {model_name}")
        ResWidget().model = Model(model_name, (112, 112, 3), None, 12)
        ResWidget().model.load()

    @staticmethod
    def train_model(model, dataset_name, input_format, lr, epochs, custom_callbacks):
        data = model_trainer.prepare_data(dataset_name, Dataset.format(input_format), (112, 112))
        ResWidget().set_batch_amount(np.ceil(len(data.get(Dataset.TRAIN))/CONST.BATCH_SIZE))
        scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
        model_trainer.train_test(model, data, lr, epochs, scheduler, callbacks=custom_callbacks)

    @staticmethod
    def resize_frames(path, shape):
        return cv2.resize(cv2.imread(path), shape, interpolation=cv2.INTER_AREA).astype("float32") / 255

    @staticmethod
    def set_batch_amount(value):
        ResWidget().batch_amount = value

    def addWidget(self, widget):
        return self.layout.addWidget(widget)

    def addLayout(self, layout):
        return self.layout.addLayout(layout)

    @staticmethod
    def pred_label():
        pred = QtWidgets.QLabel()
        pred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred.size()
        pred.setFixedHeight(15)
        return pred

    @staticmethod
    def predict(model, path):
        img = ResWidget.resize_frames(path, (112, 112))
        input_img = np.expand_dims(img, axis=0)
        [res] = model(input_img, training=False)
        res = [(CONST.keep_stanford40[idx], res.numpy()[idx]) for idx in np.argsort(res)[-3:]]
        return res

    @staticmethod
    def display_prediction(label, res, correct):
        label.setText(' '.join(res[0].split('_')).capitalize() +f': {round(res[1] * 100)}%')
        if res[0] == correct:
            label.setStyleSheet("color: #90EE90;")
        else:
            label.setStyleSheet("color: white;")
