from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Slot
from data_loader import Dataset

import cv2
import numpy as np

class ResWidget(QtWidgets.QWidget):

    def __init__(self, predict):
        super().__init__()
        self.folder_path = "datasets/Stanford40/JPEGImages/"
        # List all files in the folder
        self.index = 0

        self.predict = predict

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

        self.pics = None
        self.labels = None
        self.categories = None
        self.input_data = None
        self.is_two_stream = False
        self.batch_amount = None

    def set_pics(self, dataset):
        self.input_data = dataset.get(Dataset.TEST)
        if dataset.layout == Dataset.TWO_STREAM:
            self.pics = self.input_data[Dataset.FRAME]
            self.is_two_stream = True
        else:
            self.pics = self.input_data
        alpha = np.ones(self.pics.shape[:-1] + (1,), dtype=self.pics.dtype)
        self.pics = [cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) for pic in self.pics]
        self.pics = np.concatenate((self.pics, alpha), axis=-1)
        self.labels = [dataset.action_categories[i] for i in np.argmax(dataset.tst_labels, axis=1)]
        self.categories = dataset.action_categories

    @Slot()
    def next(self):
        self.index = (self.index + 1) % len(self.pics)
        self.setPic()
        return

    @Slot()
    def prev(self):
        self.index = (self.index - 1) % len(self.pics)
        self.setPic()
        return

    def setPic(self):
        self.pixmap = QtGui.QPixmap(QtGui.QImage(self.pics[self.index], 112, 112, QtGui.QImage.Format.Format_RGBA32FPx4))
        self.pixmap = self.pixmap.scaled(300, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        category = self.labels[self.index]
        if self.is_two_stream:
            res = self.predict([np.expand_dims(self.input_data[0][self.index], axis=0), np.expand_dims(self.input_data[1][self.index], axis=0)], self.categories)
        else:
            res = self.predict(np.expand_dims(self.input_data[self.index], axis=0), self.categories)

        self.display_prediction(self.pred_1, res[2], category)
        self.display_prediction(self.pred_2, res[1], category)
        self.display_prediction(self.pred_3, res[0], category)

        self.pic.setPixmap(self.pixmap)
        return

    def addWidget(self, widget):
        return self.layout.addWidget(widget)

    def addLayout(self, layout):
        return self.layout.addLayout(layout)

    def display_prediction(self, label, res, correct):
        label.setText(' '.join(res[0].split('_')).capitalize() +f': {round(res[1] * 100)}%')
        if res[0] == correct:
            label.setStyleSheet("color: #90EE90;")
        else:
            label.setStyleSheet("color: white;")

    @staticmethod
    def pred_label():
        pred = QtWidgets.QLabel()
        pred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred.size()
        pred.setFixedHeight(15)
        return pred
