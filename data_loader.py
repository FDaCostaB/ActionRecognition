import os
import cv2
import numpy as np
import constants as CONST

from glob import glob
import files_utils as futils
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow import keras

class Dataset:
    """
    A class loading data set
    """
    FRAME = 0
    OPTICAL_FLOW = 1
    TWO_STREAM = 2

    TRAIN = 0
    TEST = 1

    def __init__(self, dataset):
        self.dataset = dataset
        self.layout = None
        match dataset.lower():
            case 'hmdb51':
                cat, tr_files, tr_labels, tst_files, tst_labels = Dataset.load_hmdb51()
            case 'stanford40':
                # cat, tr_files, tr_labels, tst_files, tst_labels = Dataset.split_stanford40()
                cat, tr_files, tr_labels, tst_files, tst_labels = Dataset.load_stanford40()
            case _:
                raise FileNotFoundError("Dataset not available. Verify the spelling")
        self.action_categories = cat
        self.tr_files = tr_files
        self.tst_files = tst_files
        self.tr_labels = tr_labels
        self.tst_labels = tst_labels
        self.train = [[], []]
        self.test = [[], []]

    def get(self, split):
        if self.layout != Dataset.TWO_STREAM:
            return self.test[self.layout] if split == Dataset.TEST else self.train[self.layout]
        else :
            return self.test if split == Dataset.TEST else self.train

    def prepare(self, layout, shape, shape_opt_flow, method, frames=None, force_preprocessing=False):
        self.layout = layout
        if layout == Dataset.FRAME or layout == Dataset.TWO_STREAM:
            self.to_frames(shape, method)
        if layout == Dataset.OPTICAL_FLOW or layout == Dataset.TWO_STREAM:
            self.to_opt_flows(CONST.OPT_FLOW_PATH_TRAIN, CONST.OPT_FLOW_PATH_TEST, shape_opt_flow, frames, force_preprocessing)

        self.tr_labels = np.array([self.action_categories.index(label) for label in self.tr_labels])
        self.tst_labels = np.array([self.action_categories.index(label) for label in self.tst_labels])

        self.tr_labels = keras.utils.to_categorical(self.tr_labels, len(self.action_categories))
        self.tst_labels = keras.utils.to_categorical(self.tst_labels, len(self.action_categories))

    def to_frames(self, shape, method):
        match self.dataset.lower():
            case 'hmdb51':
                self.get_middle_frames(shape)
            case 'stanford40':
                if method.lower() == "resize":
                    self.resize_frames(shape)
                elif method.lower() == "crop":
                    self.crop_frames(shape)
                else:
                    raise NameError("Unknown method. Verify spelling")
            case _:
                raise FileNotFoundError("Dataset not available. Verify spelling")

    def get_middle_frames(self, shape):
        train_frames = []
        for i in range(len(self.tr_files)):
            video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{self.tr_labels[i]}/{self.tr_files[i]}')
            frameNr = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / 2)
            middleFrame = frameNr
            video.set(cv2.CAP_PROP_POS_FRAMES, middleFrame)
            ret, frame = video.read()
            frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
            train_frames.append(frame)
        train_frames = np.array(train_frames)
        self.train[Dataset.FRAME] = train_frames.astype("float32") / 255

        test_frames = []
        for i in range(len(self.tst_files)):
            video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{self.tst_labels[i]}/{self.tst_files[i]}')
            frameNr = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / 2)
            middleFrame = frameNr
            video.set(cv2.CAP_PROP_POS_FRAMES, middleFrame)
            ret, frame = video.read()
            frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
            test_frames.append(frame)
        test_frames = np.array(test_frames)
        self.test[Dataset.FRAME] = test_frames.astype("float32") / 255

    def crop_frames(self, shape):
        self.tr_frames = []
        for file_name in self.tr_files:
            img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
            (width, height, _) = img.shape
            if width < 224 or height < 224:
                img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            else:
                (cropw, croph) = shape
                x = np.random.randint(0, width - cropw)
                y = np.random.randint(0, height - croph)
                # Crop the image
                img = img[x:x + cropw, y:y + croph]
            self.tr_frames.append(img)
        self.tr_frames = np.array(self.tr_frames)
        self.train[Dataset.FRAME] = self.tr_frames.astype("float32") / 255

        self.tst_frames = []
        for file_name in self.tst_files:
            img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
            (width, height, _) = img.shape
            if width < 224 or height < 224:
                img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            else:
                (cropw, croph) = shape
                x = np.random.randint(0, width - cropw)
                y = np.random.randint(0, height - croph)
                # Crop the image
                img = img[x:x + cropw, y:y + croph]
            self.tst_frames.append(img)
        self.tst_frames = np.array(self.tst_frames)
        self.test[Dataset.FRAME] = self.tst_frames.astype("float32") / 255

    def resize_frames(self, shape):
        self.tr_frames = []
        for file_name in self.tr_files:
            img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
            img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            self.tr_frames.append(img)
        self.tr_frames = np.array(self.tr_frames)
        self.train[Dataset.FRAME] = self.tr_frames.astype("float32") / 255

        self.tst_frames = []
        for file_name in self.tst_files:
            img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{file_name}')
            img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            self.tst_frames.append(img)
        self.tst_frames = np.array(self.tst_frames)
        self.test[Dataset.FRAME] = self.tst_frames.astype("float32") / 255

    def to_opt_flows(self, train_path, test_path, shape, frames, force_preprocessing):
        try:
            if force_preprocessing:
                self.tr_labels, all_opt_train_flows = self.get_opt_flow_from_videos(train_path, "train", shape, frames)
                self.tst_labels, all_opt_test_flows = self.get_opt_flow_from_videos(test_path, "test", shape, frames)
            else:
                tr_optical_flows = np.load(train_path)
                ts_optical_flows = np.load(test_path)

                all_opt_train_flows = tr_optical_flows['allOpticalFlows']
                all_opt_test_flows = ts_optical_flows['allOpticalFlows']
        except FileNotFoundError:
            self.tr_labels, all_opt_train_flows = self.get_opt_flow_from_videos(train_path, "train", shape, frames)
            self.tst_labels, all_opt_test_flows = self.get_opt_flow_from_videos(test_path, "test", shape, frames)

        self.train[Dataset.OPTICAL_FLOW] = np.array(all_opt_train_flows).astype("float32")
        self.test[Dataset.OPTICAL_FLOW] = np.array(all_opt_test_flows).astype("float32")

    def get_opt_flow_from_videos(self, path, split, shape, frames):
        all_optical_flow = []
        new_labels = []
        labels = self.tst_labels if split == "test" else self.tr_labels
        files = self.tst_files if split == "test" else self.tr_files
        for i in range(len(labels)):
            opt_flow_frames = []
            video = cv2.VideoCapture(f'./datasets/HMDB51/video_data/{labels[i]}/{files[i]}')
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = int(frame_count / (frames + 2))
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = video.read()
            frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for frameNr in range(frames):
                video.set(cv2.CAP_PROP_POS_FRAMES, min((frameNr + 1) * frame_step, frame_count))
                ret, next_frame = video.read()

                next_frame = cv2.resize(next_frame, shape, interpolation=cv2.INTER_AREA)
                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                optical_flow = cv2.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                mag, dir = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
                hsv[..., 0] = dir * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                optical_flow = np.asarray(optical_flow, np.float32)
                opt_flow_frames.append(optical_flow / 255)
                frame = next_frame
            all_optical_flow.append(opt_flow_frames)
            new_labels.append(labels[i])
            print(f"{i}/{len(labels)}")
        np.savez(path, allOpticalFlows = all_optical_flow)

        return new_labels, all_optical_flow

    @staticmethod
    def split_stanford40():
        train_files, train_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/train.txt', CONST.keep_stanford40)
        test_files, test_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/test.txt', CONST.keep_stanford40)

        # Combine the splits and split for keeping more images in the training set than the test set.
        all_files = train_files + test_files
        all_labels = train_labels + test_labels
        train_files, test_files = train_test_split(all_files, test_size=CONST.STANFORD_TEST_SIZE, random_state=0, stratify=all_labels)
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

        with open("./datasets/Stanford40/test.txt", "w") as file:
            # Write each string in the list as a new line in the file
            for file_name in sorted(test_files):
                file.write(file_name + "\n")

        with open("./datasets/Stanford40/train.txt", "w") as file:
            # Write each string in the list as a new line in the file
            for file_name in sorted(train_files):
                file.write(file_name + "\n")

        action_categories = sorted(list(set(train_labels)))
        return action_categories, train_files, train_labels, test_files, test_labels

    @staticmethod
    def format(text):
        match text.lower():
            case 'frames':
                return Dataset.FRAME
            case 'optical flow':
                return Dataset.OPTICAL_FLOW
            case 'both':
                return Dataset.TWO_STREAM

    @staticmethod
    def load_stanford40():
        train_files, train_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/train.txt', CONST.keep_stanford40)
        test_files, test_labels = futils.parse_filelist_Stanford40('./datasets/Stanford40/test.txt', CONST.keep_stanford40)

        action_categories = sorted(list(set(train_labels)))
        return action_categories, train_files, train_labels, test_files, test_labels

    @staticmethod
    def load_hmdb51():
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

        # print(f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
        action_categories = sorted(list(set(train_labels)))
        return action_categories, train_files, train_labels, test_files, test_labels