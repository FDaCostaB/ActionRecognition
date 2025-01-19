BATCH_SIZE = 64
STANFORD_TEST_SIZE = 0.1
OPTICAL_FLOW_FRAMES = 10

PLOT_SHOW = False
PLOT_SAVE = True

OUTPUT_PATH = './output/'
PLOT_PATH = OUTPUT_PATH + 'plots/'
OPT_FLOW_PATH_TRAIN = OUTPUT_PATH + 'OpticalFlows/opticalFlow_train.npz'
OPT_FLOW_PATH_TEST = OUTPUT_PATH + 'OpticalFlows/opticalFlow_test.npz'

format = ["Frames", "Optical flow", "Both"]
model = ['cnn', 'cnn_2', 'deep_cnn_2', 'alightnet', 'opt_flow_cnn', 'two_stream_cnn']
dataset = ['Stanford40', 'HMDB51']

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