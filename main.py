from data_loader import Dataset
from models import Model


def train_cnn_standford40():
    model = Model("cnn", 12)

    data = Dataset("Stanford40")
    data.prepare(Dataset.FRAME)

    model.train(data, 0.0005, 50, 10)
    #model.load()
    model.test(data)


def finetune_cnn_for_hmdb51():
    model = Model("cnn_pretrained", 12)

    data = Dataset("HMDB51")
    data.prepare(Dataset.FRAME)

    #model.train(data, 0.0004, 50, 10)
    model.load()
    model.test(data)

def train_single_stream_optical():
    model = Model("opt_flow_cnn", 12)

    data = Dataset("HMDB51")
    data.prepare(Dataset.OPTICAL_FLOW, False)

    #model.train(data, 0.0005, 50, 5)
    model.load()
    model.test(data)

def train_two_stream_cnn():
    model = Model("two_stream_cnn", 12)

    data = Dataset("HMDB51")
    data.prepare(Dataset.TWO_STREAM, False)

    #model.train(data, 0.0002, 50, 5)
    model.load()
    model.test(data)


if __name__ == '__main__':
    print("Start")
    train_cnn_standford40()
    #finetune_cnn_for_hmdb51()
    #train_single_stream_optical()
    #train_two_stream_cnn()