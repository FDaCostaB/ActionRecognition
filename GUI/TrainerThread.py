from PySide6.QtCore import QThread, Signal
from custom_callback import CustomCallback
from GUI import ResWidget


class TrainerThread(QThread):
    """
    A worker thread to handle a long-running task.
    """
    progress = Signal(int)  # Signal to send progress updates to the main thread
    stopped = Signal()  # Signal emitted when the task is stopped
    finished = Signal()  # Signal emitted when the task is stopped

    def __init__(self, model, dataset, in_format, lr, epochs):
        super().__init__()
        self._is_running = True  # Flag to control the thread

        self.model = model
        self.dataset = dataset
        self.in_format = in_format
        self.lr = lr
        self.total_epochs = epochs
        self.custom_callback = CustomCallback(self.on_epoch_begin, self.on_epoch_end, self.on_batch_end)

    def run(self):
        """
        Code executed in the worker thread.
        """
        ResWidget().train_model(self.model, self.dataset, self.in_format, self.lr, self.total_epochs, self.custom_callback)
        if self._is_running:
            self.finished.emit()

    def stop(self):
        """Request the thread to stop."""
        self._is_running = False

    def on_epoch_begin(self, epoch, logs=None):
        return None
        #print(f"{round(epoch/self.total_epochs * 100)} %")
        #self.progress.emit(round(epoch/self.total_epochs * 100))

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.total_epochs} || loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.4f} - val_loss: {logs.get('val_loss'):.4f} - val_accuracy: {logs.get('val_accuracy'):.4f}")
        #print(f"{round( (epoch + 1) / self.total_epochs * 100)} %")
        #self.progress.emit(round(epoch/self.total_epochs * 100))

    def on_batch_end(self, epoch, batch, logs=None):
        if not self._is_running:
            self.stopped.emit()  # Notify that the thread was stopped
            self.model.model.stop_training = True  # This stops the training
        batch_progress = (batch + 1) / (ResWidget().batch_amount * self.total_epochs)
        epoch_progress = (epoch) / self.total_epochs
        self.progress.emit(round((epoch_progress + batch_progress) * 100))     # Current batch index (0-based)

    def is_running(self):
        return self._is_running
