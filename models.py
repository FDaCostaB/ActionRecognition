import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras import layers
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
    def __init__(self, model_name, input_shape, opt_input_shape, nb_category, do_data_augment=False):
        self.name = model_name
        match model_name.lower():
            case "cnn":
                self.model = Model.createStandford40CNN(input_shape, nb_category, do_data_augment)
            case "cnn_2":
                self.model = Model.createCNN2(input_shape, nb_category)
            case "deep_cnn_2":
                self.model = Model.createDeepCNN2(input_shape, nb_category)
            case "opt_flow_cnn":
                self.model = Model.createOpticalFlowCNN(opt_input_shape, nb_category)
            case "two_stream_cnn":
                self.model = Model.createTwoStreamCNN(input_shape, opt_input_shape, nb_category)
            case "alightnet":
                self.model = Model.createAlightNet(input_shape, nb_category)

    def train(self, data, lr, epochs, lr_scheduler, callbacks=None):
        plot_model(self.model, to_file=CONST.PLOT_PATH + self.name +".png", show_shapes=True, show_layer_names=True)

        val_data = (data.get(Dataset.TEST), data.tst_labels)
        opt = keras.optimizers.Adam(lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        cbs = [lr_scheduler.callbacks]
        if callbacks != None:
            cbs.append(callbacks)

        eval = self.model.fit(data.get(Dataset.TRAIN), data.tr_labels, batch_size=CONST.BATCH_SIZE, epochs=epochs,
                              validation_data=val_data, callbacks=[lr_scheduler.callbacks, cbs], verbose=0)

        history = [eval.history['loss'], eval.history['accuracy'], eval.history['val_loss'], eval.history['val_accuracy']]
        np.savez(CONST.OUTPUT_PATH + self.name + '_eval', history=history)
        self.model.save(CONST.OUTPUT_PATH + self.name)

    def load(self):
        self.model = load_model(CONST.OUTPUT_PATH + self.name)
        plot_model(self.model, to_file=CONST.PLOT_PATH + self.name +".png", show_shapes=True, show_layer_names=True)
        history = np.load(CONST.OUTPUT_PATH + self.name + '_eval.npz')
        history = history['history']
        if not CONST.PLOT_SAVE and not CONST.PLOT_SHOW:
            self.plot_evaluation(history, len(history[0]))

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
        pred_top3 = [pred[-3:] for pred in np.argsort(predictions, axis=1)]
        top3_acc = 0
        for i in range(len(tst_index)):
            if tst_index[i] in pred_top3[i]:
                top3_acc = top3_acc + 1
        top3_acc = top3_acc / len(tst_index)
        print("Top 3 accuracy", top3_acc)
        print("Global Loss", score[0])
        print("Global accuracy:", score[1])

    def plot_evaluation(self, history, epochs):
        plot.plotLossToEpoch(history, self.name, epochs)
        plot.plotAccToEpoch(history, self.name, epochs)

    @staticmethod
    def createStandford40CNN(input_shape, output_size, do_data_augment):
        output_layer = layers.Dense(output_size, activation="softmax")
        if do_data_augment:
            data_augmentation = Model.data_augmentation()
        else:
            data_augmentation = layers.Identity()

        stanford40_cnn = Sequential([
            keras.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(16, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(32, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(64, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dense(256, activation="swish"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="swish"),
            layers.Dropout(0.5),
            output_layer,
        ])
        return stanford40_cnn

    @staticmethod
    def createCNN2(input_shape, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")

        cnn_2 = Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(128, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(256, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dense(256, activation="swish"),
            layers.Dropout(0.6),
            layers.Dense(256, activation="swish"),
            layers.Dropout(0.6),
            output_layer,
        ])
        return cnn_2

    @staticmethod
    def createDeepCNN2(input_shape, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")

        deep_cnn_2 = Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(128, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(256, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(512, kernel_size=(5, 5), activation="swish"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dense(1024, activation="swish"),
            layers.Dropout(0.7),
            layers.Dense(128, activation="swish"),
            layers.Dropout(0.5),
            output_layer,
        ])
        return deep_cnn_2

    @staticmethod
    def createOpticalFlowCNN(opt_input_shape, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")
        (width, height) = opt_input_shape

        optical_flow_cnn = Sequential([
            keras.Input(shape=(CONST.OPTICAL_FLOW_FRAMES, width, height, 3)),
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
    def createAlightNet(input_shape, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")
        # Define the AlexNet model
        aligthnet = Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(8, kernel_size=(11, 11), activation="swish"),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),

            layers.Conv2D(16, kernel_size=(5, 5), activation="swish"),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),

            layers.Conv2D(32, kernel_size=(3, 3), activation="swish"),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),

            layers.Conv2D(32, kernel_size=(3, 3), activation="swish"),
            layers.BatchNormalization(),
            layers.Activation('swish'),

            layers.Conv2D(16, kernel_size=(5, 5), activation="swish"),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=2),

            layers.Flatten(),
            layers.Dense(256, activation="swish"),
            # layers.BatchNormalization(),
            # layers.Activation('swish'),
            layers.Dropout(0.5),

            layers.Dense(256, activation="swish"),
            # layers.BatchNormalization(),
            # layers.Activation('swish'),
            layers.Dropout(0.5),

            output_layer,
        ])
        return aligthnet

    @staticmethod
    def createStandford40TwoStreamCNN(input_shape):
        stanford40_cnn = Sequential([
            keras.Input(shape=input_shape),
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
    def createTwoStreamOpticalCNN(input_shape):
        optical_flow_cnn = Sequential([
            keras.Input(shape=input_shape),
            layers.Conv3D(16, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(2, 2, 1)),
            layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Conv3D(32, kernel_size=(2, 3, 3), activation="relu"),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
        ])

        return optical_flow_cnn


    @staticmethod
    def createTwoStreamCNN(input_shape, opt_input_shape, output_size):
        output_layer = layers.Dense(output_size, activation="softmax")

        model_optical = Model.createTwoStreamOpticalCNN(opt_input_shape)
        trained_model_optical = load_model(CONST.OUTPUT_PATH + "opt_flow_cnn")

        model_frame = Model.createStandford40TwoStreamCNN(input_shape)
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
    def data_augmentation():
        data_augmentation = Sequential([
            layers.RandomTranslation(0.1,0.1),
            layers.RandomFlip("horizontal")
        ])
        return data_augmentation