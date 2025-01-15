import constants as CONST
import matplotlib.pyplot as plt
import numpy as np

LOSS = 0
ACC = 1
VAL_LOSS = 2
VAL_ACC = 3

def plotLossToEpoch(history, modelName, epochs):
    train_loss = history[LOSS]
    val_loss = history[VAL_LOSS]
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss)
    plt.plot(range(1, len(val_loss) + 1), val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 4)
    plt.xlim(1, epochs)
    plt.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.legend(['train', 'validation'], loc='upper left')
    if CONST.PLOT_SAVE:
        plt.savefig(CONST.PLOT_PATH + modelName + "_loss.png")
    if CONST.PLOT_SHOW:
        plt.show()

def plotAccToEpoch(history, modelName, epochs):
    train_acc = history[ACC]
    val_acc = history[VAL_ACC]
    plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc)
    plt.plot(range(1, len(val_acc) + 1), val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(1, epochs)
    plt.ylim(0, 1)
    plt.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(['train', 'validation'], loc='upper left')
    if CONST.PLOT_SAVE:
        plt.savefig(CONST.PLOT_PATH + modelName + "_acc.png")
    if CONST.PLOT_SHOW:
        plt.show()