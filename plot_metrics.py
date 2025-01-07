import constants as CONST
import matplotlib.pyplot as plt

LOSS = 0
ACC = 1
VAL_LOSS = 2
VAL_ACC = 3

def plotLossToEpoch(history, modelName):
    train_loss = history[LOSS]
    val_loss = history[VAL_LOSS]
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 50)
    plt.legend(['train', 'validation'], loc='upper left')
    if CONST.PLOT_SAVE:
        plt.savefig(CONST.PLOT_PATH + modelName + "_loss.png")
    if CONST.PLOT_SHOW:
        plt.show()

def plotAccToEpoch(history, modelName):
    train_acc = history[ACC]
    val_acc = history[VAL_ACC]
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.legend(['train', 'validation'], loc='upper left')
    if CONST.PLOT_SAVE:
        plt.savefig(CONST.PLOT_PATH + modelName + "_acc.png")
    if CONST.PLOT_SHOW:
        plt.show()