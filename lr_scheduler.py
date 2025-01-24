import numpy as np
import constants as CONST
from keras.src.callbacks import ReduceLROnPlateau, LearningRateScheduler


class lrScheduler:

    SMOOTH_DECAY = 0
    FACTOR = 1
    REDUCE_PLATEAU = 2
    TRIANGULAR = 3

    def __init__(self, method, decay=10, factor=0.5, patience=3, base_lr=5e-4, monitor='val_loss'):
        match method:
            case self.SMOOTH_DECAY:
                self.method = lambda epoch, lr: lrScheduler.learningRateSchedulerDecreasing(epoch, lr, decay)
                self.callbacks = LearningRateScheduler(self.method, verbose=0)
                return
            case self.FACTOR:
                self.method = lambda epoch, lr: lrScheduler.lrSchedulerFactor(epoch, lr, decay, factor)
                self.callbacks = LearningRateScheduler(self.method, verbose=0)
                return
            case self.REDUCE_PLATEAU:
                self.callbacks = [ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=1e-6, min_delta=1e-4, verbose=0)]
                return
            case self.TRIANGULAR:
                self.method = lambda epoch, lr: lrScheduler.lrSchedulerTriangular(epoch, factor, base_lr, decay)
                self.callbacks = LearningRateScheduler(self.method, verbose=0)
                return

    @staticmethod
    def learningRateSchedulerDecreasing(epoch, lr, decay):
        new_lr = lr
        if epoch % decay == 0 and epoch:
            new_lr = lr - (lr / ((epoch / decay) + 1))
        return new_lr

    @staticmethod
    def lrSchedulerFactor(epoch, lr, decay, factor):
        new_lr = lr
        if epoch % decay == 0 and epoch:
            new_lr = lr * factor
        return new_lr

    @staticmethod
    def lrSchedulerTriangular(epoch, factor, base_lr, decay):
        cycle = 1 + np.floor(epoch / decay)
        x = np.abs(epoch / (decay * 0.5) - 2 * cycle + 1)

        scale = factor ** (cycle - 1)  # Shrinks max_lr every cycle

        new_lr = base_lr + base_lr * max(0, (1 - x)) * scale
        return new_lr

    @staticmethod
    def FromText(method):
        return CONST.scheduler_types.index(method)
