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
from sklearn.model_selection import KFold
from tqdm import tqdm
import bcolz
import random as rnd
import cv2

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
    NAME = "airbus_chip"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 6

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

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

    LEARNING_RATE = 0.001


class AirbusInferenceConfig(AirbusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    IMAGES_PER_GPU = 4

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.99

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3



############################################################
#  Dataset
############################################################

class AirbusShipDataset(utils.Dataset):

    def load_airbus_ship(self, data):
        self.add_class("ship", 1, "ship")

        self.chips = bcolz.open("chips", mode='r')

        for idname, ipath, codes in data:

            chip_indexes = []
            chip_boxes = []
            if str(codes[0]) == "nan":

                number_of_chips_to_add = 15

                partition_index = rnd.randint(0, len(self.chips.partitions) - 1)
                chip_indexes = rnd.sample(range(self.chips.partitions[partition_index][0],
                                                self.chips.partitions[partition_index][1]), number_of_chips_to_add)

                chips_to_add = self.chips[chip_indexes]

                for chip_item in chips_to_add:
                    x1, x2, y1, y2 = chip_item["x1"], chip_item["x2"], chip_item["y1"], chip_item["y2"]

                    chip_width = x2 - x1
                    chip_height = y2 - y1

                    loop_counter = 0
                    while loop_counter < 100:

                        x_start = rnd.randint(0, 768 - chip_width)
                        y_start = rnd.randint(0, 768 - chip_height)

                        rect_a = [x_start, x_start + chip_width, y_start, y_start + chip_height]

                        if len([rect_b for rect_b in chip_boxes if self.overlap(rect_a, rect_b)]) == 0:
                            chip_boxes.append(rect_a)
                            break
                        loop_counter += 1


            self.add_image(
                "ship",
                image_id=idname,
                path=ipath, 
                codes=codes,
                chip_indexes=chip_indexes,
                chip_boxes=chip_boxes)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        shape = (768, 768)

        masks = self.codes2masks(image_info["codes"], shape)
        masks = np.moveaxis(masks, 0, -1)

        if masks.max() < 1:
            masks = []

        for chip_item, chip_box in zip(self.chips[image_info["chip_indexes"]], image_info["chip_boxes"]):
            chip_mask = chip_item["mask"]
            (x_start, x_end, y_start, y_end) = chip_box

            new_mask = np.zeros((768, 768))
            new_mask[y_start: y_end, x_start: x_end] = chip_mask

            masks.append(new_mask)

        if type(masks) is list:
            masks = np.moveaxis(np.array(masks), 0, -1)

        if masks.max() < 1:
            return masks, np.zeros_like([masks.shape[-1]], dtype=np.int32)
        else:
            return masks, np.ones([masks.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image

        image_info = self.image_info[image_id]
        path = image_info['path']
        image = skimage.io.imread(path)
        #image = cv2.imread(path)[:, :, (2, 1, 0)]

        for chip_item, chip_box in zip(self.chips[image_info["chip_indexes"]], image_info["chip_boxes"]):
            chip_mask = chip_item["mask"]
            chip = chip_item["image"]
            (x_start, x_end, y_start, y_end) = chip_box

            chip_height = y_end - y_start
            chip_width = x_end - x_start

            chip_mask_inv = cv2.bitwise_not(chip_mask) - 254
            chip_mask_rgb = cv2.bitwise_and(chip, chip, mask=chip_mask)

            river = image[y_start: y_start + chip_height, x_start: x_start + chip_width]
            river_mask = cv2.bitwise_and(river, river, mask=chip_mask_inv)

            result = cv2.add(river_mask, chip_mask_rgb)
            image[y_start: y_start + chip_height, x_start: x_start + chip_width] = result


        return image

    def overlap(self, rect_a, rect_b):

        def range_overlap(a_min, a_max, b_min, b_max):
            '''
            Neither range is completely greater than the other
            '''
            return (a_min <= b_max) and (b_min <= a_max)

        x1_a, x2_a, y1_a, y2_a = rect_a
        x1_b, x2_b, y1_b, y2_b = rect_b

        return range_overlap(x1_a, x2_a, x1_b, x2_b) \
               and range_overlap(y1_a, y2_a, y1_b, y2_b)

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        if not isinstance(mask_rle, str):
            return np.zeros(shape)

        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def codes2masks(self, codes, shape):
        masks = []
        for code in codes:
            masks.append(self.rle_decode(code, shape))
        masks = np.stack(masks, axis=0)
        return masks

############################################################
#  Training
############################################################

def train(model, data, config):
    """Train the model."""
    # Training dataset.
    train_data = data[:int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9):]

    #remove empty images
    train_data = [row for row in train_data if str(row[2][0]) != "nan"]
    #val_data = [row for row in val_data if str(row[2][0]) != "nan"]

    # remove images without ships from val and add it to training
    train_data = [row for row in val_data if str(row[2][0]) == "nan"][:1000]
    val_data = [row for row in val_data if str(row[2][0]) != "nan"]

    dataset_train = AirbusShipDataset()
    dataset_train.load_airbus_ship(train_data)
    # Must call before using the dataset
    dataset_train.prepare()

    # Load and display random samples# Load

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

        plt.savefig("temp/{}.png".format(image_id))

    
    dataset_val = AirbusShipDataset()
    dataset_val.load_airbus_ship(val_data)
    # Must call before using the dataset
    dataset_val.prepare()
    
    
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        #iaa.Multiply((0.8, 1.5)),
        #iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='5+')


############################################################
#  Detection
############################################################

def detect(model, data, config):

    csv_results  = {
                    #0.9: None,
                    #0.95: None,
                    0.99: None,
                    0.995: None,
                    0.999: None}

    for key, value in csv_results.items():

        out_pred_rows = []
        for id, path, codes in tqdm(data):
            out_pred_rows += [{'ImageId': id, 'EncodedPixels': None}]

        csv_results[key] = out_pred_rows


    chunk_size = config.IMAGES_PER_GPU
    for i in tqdm(range(0, len(data), chunk_size)):

        sub_data = data[i:min(i + chunk_size, len(data))]
        images = [skimage.io.imread(path) for id, path, codes in sub_data]

        model.config.BATCH_SIZE = len(images)
        result_batch = model.detect(images, verbose=0)

        index = 0
        for data_element, r in zip(sub_data, result_batch):

            for min_score, out_pred_rows in csv_results.items():

                score_indexes = np.where(r["scores"] > min_score)[0]

                if len(score_indexes) == 0:
                    continue

                masks = r["masks"][:,:,score_indexes]
                scores = r["scores"][score_indexes]
                rois = r["rois"][score_indexes]
                class_ids = r["class_ids"][score_indexes]

                id, path, codes = data_element
                lines = mask_to_rle(id, masks, scores)

                duplicates = [row for row in out_pred_rows if row["ImageId"] == id and row['EncodedPixels'] == None]
                for duplicate in duplicates:
                    out_pred_rows.remove(duplicate)

                for line in lines:
                    out_pred_rows += [{'ImageId': id, 'EncodedPixels': line.split(',')[1]}]

                if False and len(lines[0].split(',')[1]) > 1:
                    visualize.display_instances(
                        images[index], rois, masks, class_ids,
                        ['bg', 'ship'], scores,
                        show_bbox=False, show_mask=False,
                        title="Predictions")
                    plt.savefig("../../{}/{}_{}.png".format("results",str(min_score).replace('.',"_"), id.split('.')[0]))


                submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
                submission_df.to_csv('{0}.csv'.format(str(min_score).replace('.',"_") ), index=False)

                csv_results[min_score] = out_pred_rows

            index += 1

        
############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))
    
def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return ["{},".format(image_id)]
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return lines
    

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    basic_path = args.dataset

    if args.command == "train": 
        image_path = os.path.join(basic_path, 'train')
        file_masks = pd.read_csv(os.path.join(basic_path, 'train_ship_segmentations.csv'))

        img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

        file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))]));
        for i, code in file_masks.values:
            file_data[i].append(code)

        train_data = [(i.split('.')[0], os.path.join(image_path, '{}'.format(i)), code) for i, code in file_data.items()]

    else:
        image_path = os.path.join(basic_path, 'test')
        test_image_paths = os.listdir(image_path)
        test_data = [(path, os.path.join(image_path, '{}'.format(path)), [np.nan]) for path in
                     os.listdir(image_path)]
        
        
    
    # Configurations
    if args.command == "train":
        config = AirbusConfig()
    else:
        config = AirbusInferenceConfig()
    config.display()


    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, train_data, config)
    elif args.command == "detect":
        detect(model, test_data, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

