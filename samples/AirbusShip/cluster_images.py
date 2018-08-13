import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from sklearn.cluster.k_means_ import KMeans
from tqdm import tqdm
from sklearn.externals import joblib
from shutil import copyfile

basic_path = "/data/Datasets/AirbusShipDetectionChallenge/"

image_path = os.path.join(basic_path, 'train')
file_masks = pd.read_csv(os.path.join(basic_path, 'train_ship_segmentations.csv'))

img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))]));
for i, code in file_masks.values:
    file_data[i].append(code)

data = [(i.split('.')[0], os.path.join(image_path, '{}'.format(i)), code) for i, code in file_data.items()]

data_no_ships = [row for row in data if str(row[2][0]) == "nan" and row[0] != "6384c3e78"]


resnet_feature_list = np.load("features.npy")
kmeans = joblib.load('KMeans.pkl')

class_ids = kmeans.predict(resnet_feature_list)
for ship, class_id in tqdm(zip(data_no_ships, class_ids)):

    idname, ipath, codes = ship

    copyfile(ipath, "clusters/{}/{}.jpg".format(class_id, idname))









