import os
import constants as CONST
import files_utils as futils
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Slot
from model_handler import ModelHandler



class ResWidget(QtWidgets.QWidget):

    def __init__(self, predict):
        super().__init__()
        self.folder_path = "datasets/Stanford40/JPEGImages/"
        # List all files in the folder
        self.files, _ = futils.parse_filelist_Stanford40("./datasets/Stanford40/test.txt", CONST.keep_stanford40)
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

        self.batch_amount = None

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
        res = self.predict(str(os.path.join(self.folder_path, self.files[self.index])))

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
