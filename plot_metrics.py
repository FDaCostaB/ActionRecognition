import constants as CONST
import matplotlib.pyplot as plt

def plotLossAndAcurracy(train_loss, train_acc, modelName):
    plt.figure()
    plt.xlim((0, 2.5))
    plt.ylim((0, 2.5))
    colors = []
    for i in range (len(train_loss)):
        colors.append(CONST.COLORS[i])
    plt.scatter(train_acc, train_loss, c=colors)
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.savefig(CONST.PLOT_PATH + modelName + "_lossAcc.png")
    if CONST.PLOT_SHOW:
        plt.show()

def plotLossToEpoch(train_loss, val_loss, modelName):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(CONST.PLOT_PATH + modelName + "_validation_loss.png")
    if CONST.PLOT_SHOW:
        plt.show()

def plotAccToEpoch(train_acc, val_acc, modelName):
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(CONST.PLOT_PATH + modelName + "_validation_acc.png")
    if CONST.PLOT_SHOW:
        plt.show()