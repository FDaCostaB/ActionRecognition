import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, epoch_begin, epoch_end, batch_end):
        self.epoch_begin = epoch_begin
        self.epoch_end = epoch_end
        self.batch_end = batch_end
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end(epoch, logs)

    def on_batch_end(self, batch, logs=None):
        self.batch_end(self.epoch, batch, logs)  # Current batch index (0-based)
