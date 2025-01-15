import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, LearningRateScheduler


class lrScheduler:

    SMOOTH_DECAY = 0
    FACTOR = 1
    REDUCE_PLATEAU = 2
    TRIANGULAR = 3

    def __init__(self, method, decay=None, factor=0.5, patience=3 ,base_lr=5e-4, max_lr=1e-3, monitor='val_loss'):
        match method:
            case self.SMOOTH_DECAY:
                self.method = lambda epoch, lr: lrScheduler.learningRateSchedulerDecreasing(epoch, lr, decay)
                self.callbacks = LearningRateScheduler(self.method, verbose=1)
            case self.FACTOR:
                self.method = lambda epoch, lr: lrScheduler.lrSchedulerFactor(epoch, lr, decay, factor)
                self.callbacks = LearningRateScheduler(self.method, verbose=1)
            case self.REDUCE_PLATEAU:
                self.callbacks = [ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=1e-6, min_delta=1e-4)]
            case self.TRIANGULAR:
                self.method = lambda epoch, lr: lrScheduler.lrSchedulerTriangular(epoch, lr, factor, base_lr, max_lr, 10)
                self.callbacks = LearningRateScheduler(self.method, verbose=1)

    @staticmethod
    def learningRateSchedulerDecreasing(epoch, lr, decay):
        if epoch % decay == 0 and epoch:
            return lr - (lr / ((epoch / decay) + 1))
        return lr

    @staticmethod
    def lrSchedulerFactor(epoch, lr, decay, factor):
        if epoch % decay == 0 and epoch:
            return lr * factor
        return lr

    @staticmethod
    def lrSchedulerTriangular(epoch, lr, factor, base_lr, max_lr, decay):
        cycle = 1 + np.floor(epoch / decay)
        x = np.abs(epoch / (decay * 0.5) - 2 * cycle + 1)

        scale = factor ** (cycle - 1)  # Shrinks max_lr every cycle

        new_lr = base_lr + (max_lr - base_lr) * max(0, (1 - x)) * scale
        return new_lr