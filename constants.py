INPUT_SHAPE = (112, 112, 3)
OPTICAL_FLOW_FRAMES = 10
INPUT_OPTICAL_FLOW = (OPTICAL_FLOW_FRAMES, 48, 64, 3)

TARGET_SHAPE_OPTICAL_FLOW = (64, 48)
TARGET_SHAPE = (112, 112)
INPUT_SHAPE_OPTICAL_FLOW = (64, 48, 3)

FORCE_TRAINING = False
DATA_AUGMENTATION = False
BATCH_SIZE = 64
EPOCHS = 100
DECAY_STEPS = 5
STARTING_LRATE = 0.0001
PRE_TRAIN_LEARNING_RATE = 0.003

CYCLIC_LEARNING_RATE = False
MAXIMUM_LEARNING_RATE = 0.003

FORCE_OPTICAL_FLOW = False
OPTICAL_FLOW_PATH_TRAIN = './data/opticalFlow_train.npz'
OPTICAL_FLOW_PATH_TEST = './data/opticalFlow_test.npz'

OUTPUT_PATH = './output/'
PLOT_PATH = OUTPUT_PATH + 'plots/'
PLOT_SHOW = False

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
               "run", "shoot_bow", "smoke", "throw", "wave"]

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid",
                   "riding_a_bike", "riding_a_horse", "running", "shooting_an_arrow",
                   "smoking", "throwing_frisby", "waving_hands"]

LOSS = 0
ACC = 1
VAL_LOSS = 2
VAL_ACC = 3

COLORS = [[1., 0., 0.], [1., 0.10980392, 0.], [1., 0.21960784, 0.], [1., 0.33333333, 0.], [1., 0.44313725, 0.],
          [1., 0.55294118, 0.], [1., 0.67058824, 0.], [1., 0.77647059, 0.], [1., 0.88627451, 0.], [1., 1., 0.],
          [1., 1., 0.], [0.88627451, 1., 0.], [0.77647059, 1., 0.], [0.67058824, 1., 0.], [0.55294118, 1., 0.],
          [0.44313725, 1., 0.], [0.33333333, 1., 0.], [0.21960784, 1., 0.], [0.10980392, 1., 0.], [0., 1., 0.],
          [0., 1., 0.], [0., 1., 0.10980392], [0., 1., 0.21960784], [0., 1., 0.33333333], [0., 1., 0.44313725],
          [0., 1., 0.55294118], [0., 1., 0.67058824], [0., 1., 0.77647059], [0., 1., 0.88627451], [0., 1., 1.],
          [0., 1., 1.], [0., 0.88627451, 1.], [0., 0.77647059, 1.], [0., 0.67058824, 1.], [0., 0.55294118, 1.],
          [0., 0.44313725, 1.], [0., 0.33333333, 1.], [0., 0.21960784, 1.], [0., 0.10980392, 1.], [0., 0., 1.]]