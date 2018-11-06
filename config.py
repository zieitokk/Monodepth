"""
Config for digging into self-supervised depth estimation.
Its backbone is resnet18. Its dimensionality is 128 x 416.

"""
class config(object):
    NAME = "DISSMDE"
    BACKBONE = "resnet18"
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 512
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_EPOCHS = 50
    ALPHA_IMAGE_LOSS = 0.85
    DISP_GRADIENT_LOSS_WEIGHT = 1000
    LR_LOSS = 1
    BY_NAME = True
    DO_STEREO = True
    WARP_MODE = "border"
    FILE_PATH = "/media/xiangtao/data/KITTI/data_scene_flow/training/image_2"
    DISPARITY_PATH = "/media/xiangtao/data/KITTI/data_scene_flow/training/disp_noc_0"
    def __init__(self):
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")