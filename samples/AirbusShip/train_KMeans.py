import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from sklearn.cluster.k_means_ import KMeans
from tqdm import tqdm

basic_path = "/data/Datasets/AirbusShipDetectionChallenge/"


from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input



image_path = os.path.join(basic_path, 'train')
file_masks = pd.read_csv(os.path.join(basic_path, 'train_ship_segmentations.csv'))

img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))]));
for i, code in file_masks.values:
    file_data[i].append(code)

data = [(i.split('.')[0], os.path.join(image_path, '{}'.format(i)), code) for i, code in file_data.items()]

data_no_ships = [row for row in data if str(row[2][0]) == "nan"]


resnet_feature_list = []

model = ResNet50(weights='imagenet', include_top=False)
model.summary()

for idname, ipath, codes in tqdm(data_no_ships):

    try:
        img = image.load_img(ipath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)


        features = model.predict(img_data)
        features = np.array(features)
        resnet_feature_list.append(features.flatten())
    except:
        print(ipath)


resnet_feature_list = np.array(resnet_feature_list)
np.save("features.npy", resnet_feature_list)

kmeans = KMeans(n_clusters=10, random_state=0).fit(resnet_feature_list)


from sklearn.externals import joblib
joblib.dump(kmeans, 'filename.pkl')




