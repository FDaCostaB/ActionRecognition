from lr_scheduler import lrScheduler
from data_loader import Dataset
from models import Model


def prepare_data(dataset_name, layout, shape=None, shape_opt=None, frames=None, method="resize"):
    dataset = Dataset(dataset_name)
    dataset.prepare(layout, shape, shape_opt, method, frames)
    return dataset


def train_test(model, dataset, lr, epoch, lr_scheduler, callbacks=None, train=True):
    if train:
        model.train(dataset, lr, epoch, lr_scheduler, callbacks=callbacks)
    else:
        model.load()
    model.test(dataset)


if __name__ == '__main__':
    # scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 10)
    # data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
    # train_test("cnn", data, 0.0005, 30, (112, 112, 3), scheduler)

    # scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 10)
    # data = prepare_data("HMDB51", Dataset.FRAME, (112, 112))
    # train_test("cnn_pretrained", data, 0.0004, 30, (112, 112, 3), scheduler)

    data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
    scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
    model = Model("cnn_2", (112, 112, 3), None, 12)
    train_test(model, data, 0.0006, 30, scheduler) # ~45-50%

    # data = prepare_data("Stanford40", Dataset.FRAME, (112, 112))
    # scheduler = lrScheduler(lrScheduler.REDUCE_PLATEAU, factor=0.33, patience=2)
    # train_test("deep_cnn_2", data, 0.0006, 30, (112, 112, 3), scheduler) # 48~50%

    # scheduler = lrScheduler(lrScheduler.DIVIDE_TEN, 15)
    # data = prepare_data("Stanford40", Dataset.FRAME, (224, 224))
    # train_test("alightnet", "Stanford40", data, 0.01, 60, (224, 224, 3), scheduler)

    # scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 5)
    # data = prepare_data("HMDB51", Dataset.OPTICAL_FLOW, shape_opt=(64, 48), frames=10)
    # train_test("opt_flow_cnn", data, 0.0005, 50, (112, 112, 3), scheduler, (64, 48, 3))

    # scheduler = lrScheduler(lrScheduler.SMOOTH_DECAY, 5)
    # data = prepare_data("HMDB51", Dataset.TWO_STREAM, (112, 112), (64, 48), 10)
    # train_test("two_stream_cnn", data, 0.0002, 50, (112, 112, 3), scheduler, (64, 48, 3))



