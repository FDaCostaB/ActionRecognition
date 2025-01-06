import constants as CONST
import cv2

def parse_filelist_Stanford40(to_parse_file, keep_category):
    """
    Parse a file containing a list of filename for ML training from parse_filelist_Stanford40 dataset

    Args:
        to_parse_file (file): file containing the list of file to train/test with (according to keep_category).

    Returns:
        list: the list of files
        list: the list of labels
    """
    with open(to_parse_file, 'r') as f:
        # We won't use these splits but split them ourselves
        files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_category]
        labels = ['_'.join(name.split('_')[:-1]) for name in files]
    return files, labels


def showFrame(image_no, category=CONST.keep_stanford40):
    train_files, train_labels = parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/train.txt', [category])
    test_files, test_labels = parse_filelist_Stanford40('./datasets/Stanford40/ImageSplits/test.txt', [category])

    # Combine the splits and split for keeping more images in the training set than the test set.
    all_files = train_files + test_files
    all_labels = train_labels + test_labels

    image_no = image_no % len(all_files)

    img = cv2.imread(f'./datasets/Stanford40/JPEGImages/{all_files[image_no]}')
    print(f'An image with the label - {all_labels[image_no]}')
    cv2.imshow('name', img)
    cv2.waitKey(0)