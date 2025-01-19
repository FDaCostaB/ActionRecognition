import numpy as np
from PySide6.QtCore import QThread, Signal
from custom_callback import CustomCallback
import constants as CONST
from data_loader import Dataset


class TrainerThread(QThread):
    """
    A worker thread to handle a long-running task.
    """
    progress = Signal(int)  # Signal to send progress updates to the main thread
    stopped = Signal()  # Signal emitted when the task is stopped
    train_ended = Signal()  # Signal emitted when the task is stopped

    def __init__(self, do_train, model, dataset_name, format, lr=None, epochs=None, scheduler=None):
        super().__init__()
        self._is_running = True  # Flag to control the thread

        self.model = model
        self.data = None
        self.do_train = do_train
        self.dataset_name = dataset_name
        self.in_format = format
        self.lr = lr
        self.total_epochs = epochs
        self.scheduler = scheduler
        self.custom_callback = CustomCallback(self.on_epoch_begin, self.on_epoch_end, self.on_batch_end)

    def run(self):
        """
        Code executed in the worker thread.
        """
        self.data = self.prepare_data(self.dataset_name, self.in_format, (112, 112), (64, 48))
        if self.do_train:
            self.model.train(self.data, self.lr, self.total_epochs, self.scheduler, self.custom_callback)
        else:
            self.model.load()
            self.model.test(self.data)
        if self._is_running:
            self.train_ended.emit()

    def stop(self):
        """Request the thread to stop."""
        self._is_running = False

    def on_epoch_begin(self, epoch, logs=None):
        return None

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.total_epochs} || loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.4f} - val_loss: {logs.get('val_loss'):.4f} - val_accuracy: {logs.get('val_accuracy'):.4f}")

    def on_batch_end(self, epoch, batch, logs=None):
        if not self._is_running:
            self.stopped.emit()  # Notify that the thread was stopped
            self.model.model.stop_training = True  # This stops the training
            return
        batch_progress = (batch + 1) / (self.get_batch_amount() * self.total_epochs)
        epoch_progress = epoch / self.total_epochs
        self.progress.emit(round((epoch_progress + batch_progress) * 100))     # Current batch index (0-based)

    def get_batch_amount(self):
        if self.in_format != Dataset.TWO_STREAM:
            return np.ceil(len(self.data.get(Dataset.TRAIN)) / CONST.BATCH_SIZE)
        else:
            return np.ceil(len(self.data.get(Dataset.TRAIN)[0]) / CONST.BATCH_SIZE)

    def is_running(self):
        return self._is_running

    @staticmethod
    def prepare_data(dataset_name, layout, shape=None, shape_opt=None, frames=None, method="resize"):
        dataset = Dataset(dataset_name)
        dataset.prepare(layout, shape, shape_opt, method, frames)
        return dataset
