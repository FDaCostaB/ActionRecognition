import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
import constants as CONST
from GUI.TrainerThread import TrainerThread
from lr_scheduler import lrScheduler
from data_loader import Dataset
from models import Model


class ModelHandler(QObject):
    """
    A worker thread to handle a long-running task.
    """
    progress = Signal(int)  # Signal to send progress updates to the main thread
    stopped = Signal()  # Signal emitted when the task is stopped
    on_test_finish = Signal()  # Signal emitted when the task is stopped

    def __init__(self):
        super().__init__()

        self.lr = None
        self.epochs = None
        self.scheduler = None
        self.params_initialized = False

        self.model = None
        self.model_name = None
        self.model_initialized = False

        self.dataset_name = None
        self.in_format = None
        self.data = None
        self.data_initialized = False

        self.trainer_thread = None

    def set_data(self, dataset_name, in_format):
        print(f"Dataset - {dataset_name} - {in_format}")
        self.dataset_name = dataset_name
        self.in_format = Dataset.format(in_format)
        self.data_initialized = True

    def set_learning_params(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        print(f"Training - lr: {lr} - epochs: {epochs}")
        self.params_initialized = True

    def init_model(self, model_name=None):
        if model_name is not None:
            self.model_name = model_name
        print(f"Model initialized : {self.model_name}")
        self.model = Model(self.model_name, (112, 112, 3), (64, 48, 3), 12)
        self.model_initialized = True

    def train_model(self):
        if self._is_trainable():

            scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
            self.trainer_thread = TrainerThread(True, self.model, self.dataset_name, self.in_format, self.lr, self.epochs, scheduler)

            print(f"epochs : {self.epochs} - lr : {self.lr} - format : {self.in_format} - Dataset : {self.dataset_name}")
            self.trainer_thread.progress.connect(self.update_progress)
            self.trainer_thread.stopped.connect(self.task_stopped)
            self.trainer_thread.train_ended.connect(self.train_finished)
            # self.trainer_thread.finished.connect(lambda: print("--FINISHED--"))
            self.trainer_thread.start()
        else:
            raise ValueError("Incorrect initialization")

    def update_progress(self, value):
        self.progress.emit(value)

    def task_stopped(self):
        self.stopped.emit()

    def train_finished(self):
        self.load_test_model()

    def test_finished(self):
        self.on_test_finish.emit()

    def stop_train(self):
        self.trainer_thread.stop()

    def load_test_model(self):
        if self._is_initialized():
            print(f"Model loaded : {self.model.name}")
            self.trainer_thread = TrainerThread(False, self.model, self.dataset_name, self.in_format)
            self.trainer_thread.train_ended.connect(self.test_finished)
            self.trainer_thread.start()
        else:
            raise ValueError("Incorrect initialization")

    def predict(self, path):
        img = self.resize_frames(path, (112, 112))
        input_img = np.expand_dims(img, axis=0)
        [res] = self.model.model(input_img, training=False)
        res = [(CONST.keep_stanford40[idx], res.numpy()[idx]) for idx in np.argsort(res)[-3:]]
        return res

    def _is_initialized(self):
        return self.data_initialized and self.model_initialized

    def _is_trainable(self):
        return self.data_initialized and self.model_initialized and self.params_initialized

    @staticmethod
    def resize_frames(path, shape):
        return cv2.resize(cv2.imread(path), shape, interpolation=cv2.INTER_AREA).astype("float32") / 255


# scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 10)
# data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
# train_test("cnn", data, 0.0006, 30, (112, 112, 3), scheduler)

# scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 10)
# data = prepare_data("HMDB51", Dataset.FRAME, (112, 112))
# train_test("cnn_pretrained", data, 0.0004, 30, (112, 112, 3), scheduler)

# data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
# scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
# model = Model("cnn_2", (112, 112, 3), None, 12)
# train_test(model, data, 0.0006, 30, scheduler) # ~45-50%

# data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
# scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
# train_test("deep_cnn_2", data, 0.0006, 30, (112, 112, 3), scheduler) # 48~50%

# scheduler = lrScheduler(lrScheduler.DIVIDE_TEN, 15)
# data = prepare_data("Stanford40", Dataset.FRAME, (224, 224))
# train_test("alightnet", "Stanford40", data, 0.01, 60, (224, 224, 3), scheduler)

# scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 5)
# data = prepare_data("HMDB51", Dataset.OPTICAL_FLOW, shape_opt=(64, 48), frames=10)
# train_test("opt_flow_cnn", data, 0.0005, 50, (112, 112, 3), scheduler, (64, 48, 3))

# scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 5)
# data = prepare_data("HMDB51", Dataset.TWO_STREAM, (112, 112), (64, 48), 10)
# train_test("two_stream_cnn", data, 0.0002, 50, (112, 112, 3), scheduler, (64, 48, 3))
