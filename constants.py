BATCH_SIZE = 64
TARGET_SHAPE = (112, 112)
INPUT_SHAPE = (112, 112, 3)

OPTICAL_FLOW_FRAMES = 10
TARGET_SHAPE_OPT_FLOW = (64, 48)
INPUT_SHAPE_OPTICAL_FLOW = (64, 48, 3)
INPUT_OPTICAL_FLOW = (OPTICAL_FLOW_FRAMES, 48, 64, 3)

PLOT_SHOW = False
PLOT_SAVE = False

OUTPUT_PATH = './output/'
PLOT_PATH = OUTPUT_PATH + 'plots/'
OPT_FLOW_PATH_TRAIN = OUTPUT_PATH + 'OpticalFlows/opticalFlow_train.npz'
OPT_FLOW_PATH_TEST = OUTPUT_PATH + 'OpticalFlows/opticalFlow_test.npz'

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
               "run", "shoot_bow", "smoke", "throw", "wave"]

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid",
                   "riding_a_bike", "riding_a_horse", "running", "shooting_an_arrow",
                   "smoking", "throwing_frisby", "waving_hands"]

COLORS = [[1., 0., 0.], [1., 0.10980392, 0.], [1., 0.21960784, 0.], [1., 0.33333333, 0.], [1., 0.44313725, 0.],
          [1., 0.55294118, 0.], [1., 0.67058824, 0.], [1., 0.77647059, 0.], [1., 0.88627451, 0.], [1., 1., 0.],
          [1., 1., 0.], [0.88627451, 1., 0.], [0.77647059, 1., 0.], [0.67058824, 1., 0.], [0.55294118, 1., 0.],
          [0.44313725, 1., 0.], [0.33333333, 1., 0.], [0.21960784, 1., 0.], [0.10980392, 1., 0.], [0., 1., 0.],
          [0., 1., 0.], [0., 1., 0.10980392], [0., 1., 0.21960784], [0., 1., 0.33333333], [0., 1., 0.44313725],
          [0., 1., 0.55294118], [0., 1., 0.67058824], [0., 1., 0.77647059], [0., 1., 0.88627451], [0., 1., 1.],
          [0., 1., 1.], [0., 0.88627451, 1.], [0., 0.77647059, 1.], [0., 0.67058824, 1.], [0., 0.55294118, 1.],
          [0., 0.44313725, 1.], [0., 0.33333333, 1.], [0., 0.21960784, 1.], [0., 0.10980392, 1.], [0., 0., 1.]]