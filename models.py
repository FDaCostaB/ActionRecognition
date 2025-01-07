import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras import layers
from keras.callbacks import LearningRateScheduler
from keras.layers import concatenate
from keras.models import Sequential
from keras.models import load_model

import numpy as np
import plot_metrics as plot
from data_loader import Dataset
import constants as CONST

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Model:
    def __init__(self, model_name, nb_category, do_data_augment=False):
        self.name = model_name
        match model_name.lower():
            case "cnn":
                self.model = Model.createStandford40CNN(nb_category, do_data_augment)
            case "cnn_pretrained":
                self.model = Model.createPreTrainedModel("cnn", nb_category)
            case "opt_flow_cnn":
                self.model = Model.createOpticalFlowCNN(nb_category)
            case "two_stream_cnn":
                self.model = Model.createTwoStreamCNN(nb_category)

    def train(self, data, lr, epochs, decay):
        plot_model(self.model, to_file=CONST.PLOT_PATH + self.name +".png", show_shapes=True, show_layer_names=True)

        val_data = (data.get(Dataset.TEST), data.tst_labels)

        lr_scheduler = lambda epoch, lr: Model.learningRateSchedulerDecreasing(epoch, lr, decay)
        callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
        opt = keras.optimizers.Adam(lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        eval = self.model.fit(data.get(Dataset.TRAIN), data.tr_labels, batch_size=CONST.BATCH_SIZE, epochs=epochs,
                              validation_data=val_data, callbacks=callbacks)

        history = [eval.history['loss'], eval.history['accuracy'], eval.history['val_loss'], eval.history['val_accuracy']]
        np.savez(CONST.OUTPUT_PATH + self.name + '_eval', history=history)
        self.model.save(CONST.OUTPUT_PATH + self.name)
        self.plot_evaluation(history)

    def load(self):
        self.model = load_model(CONST.OUTPUT_PATH + self.name)
        plot_model(self.model, to_file=CONST.PLOT_PATH + self.name +".png", show_shapes=True, show_layer_names=True)
        history = np.load(CONST.OUTPUT_PATH + self.name + '_eval.npz')
        history = history['history']
        self.plot_evaluation(history)

    def test(self, data):
        print("Start Testing")
        if self.name == "two_opt_flow_cnn":
            score = self.model.evaluate(data.get(Dataset.TEST), [data.tst_labels, data.tst_labels])
        else:
            score = self.model.evaluate(data.get(Dataset.TEST), data.tst_labels)
        print("Start Confusion matrix")
        predictions = self.model.predict(data.get(Dataset.TEST))
        pred_top1 = np.argmax(predictions, axis=1)
        tst_index = np.argmax(data.tst_labels, axis=1)
        conf_matrix = tf.math.confusion_matrix(labels=tst_index, predictions=pred_top1)
        print(conf_matrix)
        pred_top3 = [pred[-5:] for pred in np.argsort(predictions, axis=1)]
        top3_acc = 0
        for i in range(len(tst_index)):
            if tst_index[i] in pred_top3[i]:
                top3_acc = top3_acc + 1
        top3_acc = top3_acc / len(tst_index)
        print("Top 3 accuracy", top3_acc)
        print("Global Loss", score[0])
        print("Global accuracy:", score[1])


    def plot_evaluation(self, history):
        plot.plotLossToEpoch(history, self.name)
        plot.plotAccToEpoch(history, self.name)
        print("Trained CNN Model")

    @staticmethod
    def learningRateSchedulerDecreasing(epoch, lr, decay):
        if epoch % decay == 0 and epoch:
            return lr - (lr / ((epoch / decay) + 1))
        return lr

    @staticmethod
    def createStandford40CNN(output_size, do_data_augment):
        output_layer = layers.Dense(output_size, activation="softmax")
        if do_data_augment:
            data_augmentation = Model.data_augmentation()
        else:
            data_augmentation = layers.Identity()

        stanford40_cnn = Sequential([
            keras.Input(shape=CONST.INPUT_SHAPE),
            data_augmentation,
            layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            output_layer,
        ])
        return stanford40_cnn

    @staticmethod
    def createOpticalFlowCNN(output_size):
        output_layer = layers.Dense(output_size, activation="softmax")

        optical_flow_cnn = Sequential([
            keras.Input(shape=CONST.INPUT_OPTICAL_FLOW),
            layers.Conv3D(16, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(2, 2, 1)),
            layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Conv3D(32, kernel_size=(2, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            output_layer,
        ])
        return optical_flow_cnn

    @staticmethod
    def createStandford40TwoStreamCNN():
        stanford40_cnn = Sequential([
            keras.Input(shape=CONST.INPUT_SHAPE),
            layers.Identity(),
            layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
        ])

        return stanford40_cnn

    @staticmethod
    def createTwoStreamOpticalCNN():
        optical_flow_cnn = Sequential([
            keras.Input(shape=CONST.INPUT_OPTICAL_FLOW),
            layers.Conv3D(16, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(2, 2, 1)),
            layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Conv3D(32, kernel_size=(2, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
        ])

        return optical_flow_cnn

    @staticmethod
    def createTwoStreamCNN(output_size):
        output_layer = layers.Dense(output_size, activation="softmax")

        model_optical = Model.createTwoStreamOpticalCNN()
        trained_model_optical = load_model(CONST.OUTPUT_PATH + "opt_flow_cnn")

        model_frame = Model.createStandford40TwoStreamCNN()
        trained_model_frame = load_model(CONST.OUTPUT_PATH + "cnn")
        for i in range(len(model_optical.layers)):
            model_optical.layers[i].set_weights(trained_model_optical.layers[i].get_weights())

        for i in range(len(model_frame.layers)):
            model_frame.layers[i].set_weights(trained_model_frame.layers[i].get_weights())

        full_model = concatenate([model_frame.output, tf.reshape(model_optical.output, shape=(-1, 2, 2, 448))])
        full_model = (layers.Flatten())(full_model)
        full_model = (layers.Dense(2048, activation="relu"))(full_model)
        full_model = (layers.Dropout(0.5))(full_model)
        full_model = (layers.Dense(512, activation="relu"))(full_model)
        full_model = (layers.Dropout(0.5))(full_model)
        full_model = (layers.Dense(128, activation="relu"))(full_model)
        full_model = (layers.Dropout(0.5))(full_model)
        full_model = (output_layer)(full_model)

        model = keras.Model(inputs=[model_frame.input, model_optical.input], outputs=full_model)
        return model

    @staticmethod
    def createPreTrainedModel(model_name, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")
        model = load_model(CONST.OUTPUT_PATH + model_name)
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[len(model.layers) - 1] = output_layer
        return model

    @staticmethod
    def data_augmentation():
        data_augmentation = Sequential([
            layers.RandomTranslation(0.1,0.1),
            layers.RandomFlip("horizontal")
        ])
        return data_augmentation