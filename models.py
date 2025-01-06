import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import concatenate
from keras.models import Sequential
from keras.models import load_model
import constants as CONST

def createStandford40CNN(output_size, pretrained=""):
    output_layer = layers.Dense(output_size, activation="softmax")
    if CONST.DATA_AUGMENTATION:
        data_augmentation = do_data_augmentation()
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
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        output_layer,
    ])

    print("Created CNN Model")
    name = "stanford40CNN" + pretrained
    return stanford40_cnn, name


def createOpticalFlowCNN(output_size, pretrained=""):
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
        output_layer,
    ])

    print("Created CNN Model")
    name = "opticalFlowCNN" + pretrained
    return optical_flow_cnn, name

def createStandford40TwoStreamCNN(pretrained=""):
    if CONST.DATA_AUGMENTATION:
        data_augmentation = do_data_augmentation()
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
    ])

    print("Created CNN Model")
    name = "stanford40CNN" + pretrained
    return stanford40_cnn, name

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

    print("Created CNN Model")
    name = "opticalFlowCNNTwoStreamPart"
    return optical_flow_cnn, name

def createTwoStreamCNN(output_size):
    output_layer = layers.Dense(output_size, activation="softmax")

    model_optical, name = createTwoStreamOpticalCNN()
    trained_model_optical = load_model(CONST.OUTPUT_PATH + "opticalFlowCNN")

    model_frame, name = createStandford40TwoStreamCNN()
    trained_model_frame = load_model(CONST.OUTPUT_PATH + "Stanford40CNN")
    for i in range(len(model_optical.layers)):
        model_optical.layers[i].set_weights(trained_model_optical.layers[i].get_weights())

    for i in range(len(model_frame.layers)):
        model_frame.layers[i].set_weights(trained_model_frame.layers[i].get_weights())
    
    full_model = concatenate([model_frame.output,tf.reshape(model_optical.output, shape=(-1,2,2,448))])
    full_model = (layers.Flatten())(full_model)
    full_model = (layers.Dense(2048, activation="relu"))(full_model)
    full_model = (layers.Dropout(0.5))(full_model)
    full_model = (layers.Dense(512, activation="relu"))(full_model)
    full_model = (layers.Dropout(0.5))(full_model)
    full_model = (layers.Dense(128, activation="relu"))(full_model)
    full_model = (output_layer)(full_model)
    
    model = keras.Model(inputs=[model_frame.input, model_optical.input], outputs=full_model)
    name = "TwoStreamCNN"
    return model, name

    
def do_data_augmentation():
    data_augmentation = Sequential([
        layers.RandomTranslation(0.1,0.1),
        layers.RandomFlip("horizontal")
    ])
    return data_augmentation