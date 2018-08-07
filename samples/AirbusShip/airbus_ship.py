# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import pandas as pd


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/airbus_ship/")


############################################################
#  Configurations
############################################################


class AirbusConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 6

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    BACKBONE = "resnet50"
    
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768


############################################################
#  Dataset
############################################################

class AirbusShipDataset(utils.Dataset):

    def load_airbus_ship(self, image_path, ids, labels_csv=None):
        self.add_class("chip", 1, "chip")

        file_masks = pd.read_csv(labels_csv)
        img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

        img_ids = [id for id in img_ids if id in ids]

        file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))]));
        for i, code in file_masks.values:
            file_data[i].append(code)

        data = [(i.split('.')[0], os.path.join(image_path, '{}'.format(i)), code) for i, code in file_data.items()]

        for idname, ipath, codes in data:
            self.add_image(
                "nucleus",
                image_id=idname,
                path=ipath, 
                codes=codes)


