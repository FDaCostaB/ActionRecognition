import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from keras import layers
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
from glob import glob

import models as models
import plot_metrics as plot
import files_utils as futils
import constants as CONST

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

"""
Original file is located at
    https://colab.research.google.com/drive/11B0HAd9LDXYVLA29n_uorOGKa0neYFEQ

# Stanford 40
### Download the data
## Read the train and test splits, combine them and make better splits to help training networks easier.
"""

def loadStandford40():
    train_files, train_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/train.txt', CONST.keep_stanford40)
    test_files, test_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/test.txt', CONST.keep_stanford40)

    # Combine the splits and split for keeping more images in the training set than the test set.
    all_files = train_files + test_files
    all_labels = train_labels + test_labels
    train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
          f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
          f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
    action_categories = sorted(list(set(train_labels)))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')
    return action_categories, train_files, train_labels, test_files, test_labels

"""### Visualize a photo from the training files and also print its label"""


def loadHMDB51():
    TRAIN_TAG, TEST_TAG = 1, 2
    train_files, test_files = [], []
    train_labels, test_labels = [], []
    split_pattern_name = f"*test_split1.txt"
    split_pattern_path = os.path.join('./datasets/HMDB51', split_pattern_name)
    annotation_paths = glob(split_pattern_path)
    for filepath in annotation_paths:
        class_name = '_'.join(filepath.split('/')[-1].split('_')[:-2])
        class_name = class_name.split("\\")[1]
        with open(filepath) as fid:
            lines = fid.readlines()
        for line in lines:
            video_filename, tag_string = line.split()
            tag = int(tag_string)
            if tag == TRAIN_TAG:
                train_files.append(video_filename)
                train_labels.append(class_name)
            elif tag == TEST_TAG:
                test_files.append(video_filename)
                test_labels.append(class_name)

    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
          f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
          f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
    action_categories = sorted(list(set(train_labels)))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')
    return action_categories, train_files, train_labels, test_files, test_labels

def resizeImages(image, target_size = CONST.TARGET_SHAPE):
    img = cv2.imread('./datasets/Stanford40/JPEGImages/{image}', cv2.IMREAD_UNCHANGED)
    image = cv2.resize(img, CONST.TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    cv2.imshow('fdf',image)
    cv2.waitKey(0)
    return image

def learningRateSchedulerDecreasing(epoch, lr):
    if epoch % CONST.DECAY_STEPS == 0 and epoch:
        return lr - (lr / ((epoch / CONST.DECAY_STEPS)+1))
    return lr


def trainModel(model, model_name, train_files, categories, learning_rate, validation_data= None, metric="accuracy"):

    if CONST.FORCE_TRAINING:
        callbacks = [LearningRateScheduler(learningRateSchedulerDecreasing, verbose=1)]

        if CONST.CYCLIC_LEARNING_RATE:
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=CONST.STARTING_LRATE,
                                                      maximal_learning_rate=CONST.MAXIMUM_LEARNING_RATE,
                                                      scale_fn=lambda x:1/(2.**(x-1)),
                                                      step_size=2*CONST.BATCH_SIZE)
            callbacks = []
            opt = keras.optimizers.Adam(clr)
        else:    
            opt = keras.optimizers.Adam(learning_rate)

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[metric])

        if validation_data is not None:
            history = model.fit(train_files, categories, batch_size=CONST.BATCH_SIZE, epochs=CONST.EPOCHS, validation_data=validation_data, callbacks=callbacks)
        else:
            history = model.fit(train_files, categories, batch_size=CONST.BATCH_SIZE, epochs=CONST.EPOCHS, validation_split=0.1, callbacks=callbacks, shuffle=True) 
        
        history = [history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']]
        np.savez(CONST.OUTPUT_PATH + model_name + '_history', history=history)
        model.save(CONST.OUTPUT_PATH + model_name)
    else:
        model = load_model(CONST.OUTPUT_PATH + model_name)
        history = np.load(CONST.OUTPUT_PATH + model_name + '_history.npz')
        history = history['history']

    train_loss = history[CONST.LOSS]
    train_acc = history[CONST.ACC]
    val_loss = history[CONST.VAL_LOSS]
    val_acc = history[CONST.VAL_ACC]

    plot.plotLossToEpoch(train_loss, val_loss, model_name)
    plot.plotAccToEpoch(train_acc, val_acc, model_name)
    print("Trained CNN Model")
    return model


def createPreTrainedModel(model_name, output_size):
    output_layer = layers.Dense(output_size, activation="softmax")
    model = load_model(CONST.OUTPUT_PATH + model_name)
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[len(model.layers) -1] = output_layer
    name = model_name + "_pretrained"
    return model, name
        
def prepareData(train_files, test_files):
    train_images = []
    for file_name in train_files:
        img = keras.preprocessing.image.load_img(f'./datasets/Stanford40/JPEGImages/{file_name}')
        img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
        img = cv2.resize(img, CONST.TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        train_images.append(img)
    train_images = np.array(train_images)
    train_images = train_images.astype("float32") / 255

    test_images = []
    for file_name in test_files:
        img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
        img = cv2.resize(img, CONST.TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        test_images.append(img)
    test_images = np.array(test_images)
    test_images = test_images.astype("float32") / 255
    return train_images, test_images

def getMiddleFrames(train_files, test_files, train_labels, test_labels, frameNr = -1):
    train_frames = []
    for i in range(len(train_labels)):
        video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{train_labels[i]}/{train_files[i]}')
        frameNr = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / 2) 
        middleFrame = frameNr
        video.set(cv2.CAP_PROP_POS_FRAMES, middleFrame)
        ret, frame = video.read()
        frame = cv2.resize(frame, CONST.TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        train_frames.append(frame)
    train_frames = np.array(train_frames)
    train_frames = train_frames.astype("float32") / 255

    test_frames = []
    for i in range(len(test_labels)):
        video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{test_labels[i]}/{test_files[i]}')
        frameNr = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / 2) 
        middleFrame = frameNr
        video.set(cv2.CAP_PROP_POS_FRAMES, middleFrame)
        ret, frame = video.read()
        frame = cv2.resize(frame, CONST.TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        test_frames.append(frame)
    test_frames = np.array(test_frames)
    test_frames = test_frames.astype("float32") / 255

    return train_frames, test_frames

def GetOpticalFlowFromVideos(files, labels, path):
    all_optical_flow = []
    newlabels = []
    for i in range(len(labels)):
        opticalFlowFrames = []
        video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{labels[i]}/{files[i]}')
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(frame_count / (CONST.OPTICAL_FLOW_FRAMES + 2))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = video.read()
        frame = cv2.resize(frame, CONST.TARGET_SHAPE_OPTICAL_FLOW, interpolation=cv2.INTER_AREA)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frameNr in range(CONST.OPTICAL_FLOW_FRAMES):
            video.set(cv2.CAP_PROP_POS_FRAMES, min((frameNr+1) * frame_step, frame_count))
            ret, next_frame = video.read()

            next_frame = cv2.resize(next_frame, CONST.TARGET_SHAPE_OPTICAL_FLOW, interpolation=cv2.INTER_AREA)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            optical_flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            mag, dir = cv2.cartToPolar(optical_flow[...,0], optical_flow[...,1])
            hsv[..., 0] = dir * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            optical_flow = np.asarray(optical_flow, np.float32)
            opticalFlowFrames.append(optical_flow / 255)
            frame = next_frame

        all_optical_flow.append(opticalFlowFrames)
        newlabels.append(i)
        print(f"{i}/{len(labels)}")
    np.savez(path, allOpticalFlows = all_optical_flow)
    return newlabels, all_optical_flow

def train_cnn_standford40():
    action_categories, train_files, train_labels, test_files, test_labels = loadStandford40()
    train_images, test_images = prepareData(train_files, test_files)

    model, name = models.createStandford40CNN(len(action_categories))
    img_file = 'model_1.png'
    plot_model(model,to_file=CONST.PLOT_PATH+img_file,show_shapes=True,show_layer_names=True)

    train_class_data = np.array([action_categories.index(label) for label in train_labels])
    test_class_data = np.array([action_categories.index(label) for label in test_labels])

    train_class = keras.utils.to_categorical(train_class_data, len(action_categories))
    test_class = keras.utils.to_categorical(test_class_data, len(action_categories))

    trainedCNN = trainModel(model, name, train_images, train_class, CONST.STARTING_LRATE, validation_data=(test_images,test_class))

    print("Start Testing")
    score = trainedCNN.evaluate(test_images, test_class)
    print("Done Testing")
    print("Start Confusion matrix")
    predictions = trainedCNN.predict(test_images)
    predictions = np.argmax(predictions, axis=1)
    conf_matrix = tf.math.confusion_matrix(labels=test_class_data, predictions=predictions)
    print(conf_matrix)
    print("Done Confusion matrix")
    print("Global Loss", score[0])
    print("Global accuracy:", score[1])

def finetune_cnn_for_hmdb51():
    action_categories, train_files, train_labels, test_files, test_labels = loadHMDB51()
    model, name = createPreTrainedModel("stanford40CNN", len(action_categories))
    img_file = 'model_2.png'
    plot_model(model,to_file=CONST.PLOT_PATH+img_file,show_shapes=True,show_layer_names=True)
    train_images, test_images = getMiddleFrames(train_files, test_files, train_labels, test_labels, 0)

    train_class_data = np.array([ action_categories.index(label) for label in train_labels])
    test_class_data = np.array([ action_categories.index(label) for label in test_labels])

    train_class = keras.utils.to_categorical(train_class_data, len(action_categories))
    test_class = keras.utils.to_categorical(test_class_data, len(action_categories))

    trainedCNN = trainModel(model, name, train_images, train_class, validation_data=(test_images,test_class), learning_rate=CONST.PRE_TRAIN_LEARNING_RATE)
    
    print("Start Testing")
    score = trainedCNN.evaluate(test_images, test_class)
    print("Done Testing")
    print("Start Confusion matrix")
    predictions = trainedCNN.predict(test_images)
    predictions = np.argmax(predictions, axis=1)
    conf_matrix = tf.math.confusion_matrix(labels=test_class_data, predictions=predictions)
    print(conf_matrix)
    print("Done Confusion matrix")
    print("Global Loss", score[0])
    print("Global accuracy:", score[1])
    
def train_single_stream_optical():
    action_categories, train_files, tr_labels, test_files, ts_labels = loadHMDB51()
    model, name = models.createOpticalFlowCNN(len(action_categories))
    img_file = 'model_3.png'
    plot_model(model, to_file=CONST.PLOT_PATH+img_file, show_shapes=True, show_layer_names=True)
    if CONST.FORCE_OPTICAL_FLOW:
        train_labels, allOpticalTrainFlows = GetOpticalFlowFromVideos(train_files, tr_labels, CONST.OPTICAL_FLOW_PATH_TRAIN)
        test_labels, allOpticalTestFlows = GetOpticalFlowFromVideos(test_files, ts_labels, CONST.OPTICAL_FLOW_PATH_TEST)
    else:
        trainOpticalFlows = np.load(CONST.OPTICAL_FLOW_PATH_TRAIN)
        testOpticalFlows = np.load(CONST.OPTICAL_FLOW_PATH_TEST)

        allOpticalTrainFlows  = trainOpticalFlows['allOpticalFlows']
        allOpticalTestFlows = testOpticalFlows['allOpticalFlows']

    train_class_data = np.array([ action_categories.index(label) for label in tr_labels])
    test_class_data = np.array([ action_categories.index(label) for label in ts_labels])

    train_class = keras.utils.to_categorical(train_class_data, len(action_categories))
    test_class = keras.utils.to_categorical(test_class_data, len(action_categories))

    allOpticalTrainFlows = np.array(allOpticalTrainFlows).astype("float32")
    allOpticalTestFlows = np.array(allOpticalTestFlows).astype("float32")

    trainedCNN = trainModel(model, name, allOpticalTrainFlows, train_class, CONST.STARTING_LRATE, validation_data=(allOpticalTestFlows,test_class))
    
    print("Start Testing")
    score = trainedCNN.evaluate(allOpticalTestFlows, test_class)
    print("Done Testing")
    print("Start Confusion matrix")
    predictions = trainedCNN.predict(allOpticalTestFlows)
    predictions = np.argmax(predictions, axis=1)
    conf_matrix = tf.math.confusion_matrix(labels=test_class_data, predictions=predictions)
    print(conf_matrix)
    print("Done Confusion matrix")
    print("Global Loss", score[0])
    print("Global accuracy:", score[1])

def train_two_stream_cnn():
    action_categories, train_files, tr_labels, test_files, ts_labels = loadHMDB51()
    train_images, test_images = getMiddleFrames(train_files, test_files, tr_labels, ts_labels)

    model, name = models.createTwoStreamCNN(len(action_categories))
    img_file = 'model_4.png'
    plot_model(model, to_file=CONST.PLOT_PATH + img_file, show_shapes=True, show_layer_names=True)

    if CONST.FORCE_OPTICAL_FLOW:
        train_labels, allOpticalTrainFlows = GetOpticalFlowFromVideos(train_files, tr_labels, CONST.OPTICAL_FLOW_PATH_TRAIN)
        test_labels, allOpticalTestFlows = GetOpticalFlowFromVideos(test_files, ts_labels, CONST.OPTICAL_FLOW_PATH_TEST)
    else:
        trainOpticalFlows = np.load(CONST.OPTICAL_FLOW_PATH_TRAIN)
        testOpticalFlows = np.load(CONST.OPTICAL_FLOW_PATH_TEST)

        allOpticalTrainFlows  = trainOpticalFlows['allOpticalFlows']
        allOpticalTestFlows = testOpticalFlows['allOpticalFlows']

    train_class_data = np.array([ action_categories.index(label) for label in tr_labels])
    test_class_data = np.array([ action_categories.index(label) for label in ts_labels])

    train_class = keras.utils.to_categorical(train_class_data, len(action_categories))
    test_class = keras.utils.to_categorical(test_class_data, len(action_categories))

    allOpticalTrainFlows = np.array(allOpticalTrainFlows).astype("float32")
    allOpticalTestFlows = np.array(allOpticalTestFlows).astype("float32")

    inputTrainData = [train_images,allOpticalTrainFlows]
    inputTrainClasses = train_class

    inputTestData = [test_images, allOpticalTestFlows]

    trainedCNN = trainModel(model, name, inputTrainData, inputTrainClasses, CONST.STARTING_LRATE, (inputTestData, test_class))

    tTestClasses = [test_class,test_class]
    print("Start Testing")
    score = trainedCNN.evaluate(inputTestData, tTestClasses,verbose=1)
    print("Done Testing")
    print("Start Confusion matrix")
    predictions = trainedCNN.predict(inputTestData)
    predictions = np.argmax(predictions, axis=1)
    conf_matrix = tf.math.confusion_matrix(labels=test_class_data, predictions=predictions)
    print(conf_matrix)
    print("Done Confusion matrix")
    print("Global Loss", score[0])
    print("Global accuracy:", score[1])
    
if __name__ == '__main__':
    print("Start")
    #train_cnn_standford40()
    #finetune_cnn_for_hmdb51()
    #train_single_stream_optical()
    #train_two_stream_cnn()