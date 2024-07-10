import cv2
import sys
import os
import numpy as np

import dlib
import shutil
import imutils
import pickle

from PIL import Image
from pathlib import Path
from skimage import exposure
from skimage import feature
from imutils import face_utils

from sklearn.neighbors import KNeighborsClassifier
from functions import *


def Training():

    path = 'dataset'
    eyeFeatures = []
    dir_ids = []
    dir_labels = []
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePath:
        tempid = os.path.split(imagePath)
        rootpath = tempid[0]
        subpath = tempid[1]

        id = subpath.split('.')[0]
        label = subpath.split('.')[1]
        print(id)
        print(label)

        filelist = os.listdir(imagePath)
        print(filelist)

        imagecount = 0

        for imgfilename in filelist:
            imgfilenamewithpath = os.path.join(imagePath, imgfilename)
            print("Imagecount: ", imagecount, " Filename: ", imgfilenamewithpath)

            PIL_img_temp = Image.open(imgfilenamewithpath)
            PIL_img_RGB = PIL_img_temp.resize((250, 100), Image.NEAREST)
            PIL_img = PIL_img_RGB.convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            hist = feature.hog(img_numpy, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

            eyeFeatures.append(hist)
            dir_ids.append(int(id))

        dir_labels.append(label)

    # "train" the nearest neighbors classifier

    feature_array = np.array(eyeFeatures)

    print("[INFO] training classifier...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(feature_array, dir_ids)

    with open('train_model_data.pkl', 'wb') as f:
        pickle.dump([model, eyeFeatures, dir_ids, dir_labels], f)


if __name__ == "__main__":
    Training()