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
import pandas as pd
import numpy as np
import cv2
from mrcnn import visualize
from mrcnn import utils
from tqdm import tqdm

def rle_decode(mask_rle, shape):
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


def codes2masks(codes, shape):
    masks = []
    for code in codes:
        masks.append(rle_decode(code, shape))
    masks = np.stack(masks, axis=0)
    return masks

basic_path = "/data/Datasets/AirbusShipDetectionChallenge/"
image_path = os.path.join(basic_path, 'test')
file_masks = pd.read_csv("/home/c.marzahl@de.eu.local/ProgProjekte/Mask_RCNN/samples/AirbusShip/0_99.csv")

img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))]));
for i, code in file_masks.values:
    file_data[i].append(code)

test_data = [(i.split('.')[0], os.path.join(image_path, '{}'.format(i)), code) for i, code in file_data.items()]

shape = (768, 768)

for idname, ipath, codes in tqdm(test_data[:1000]):

    if "nan" in str(codes[0]):
        continue

    masks = codes2masks(codes, shape)
    masks = np.moveaxis(masks, 0, -1)

    boxes = utils.extract_bboxes(masks)
    image = cv2.imread(ipath)

    visualize.draw_boxes(image, boxes=boxes, refined_boxes=None,
               masks=masks)

    plt.savefig("results/{}.png".format(idname))