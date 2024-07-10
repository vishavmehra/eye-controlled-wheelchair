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

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



def EyeSampler(img, left_or_right_eye):


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # print(rects)

    roi = None
    if len(rects):

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            if left_or_right_eye == "left_eye":
                # extract the ROI of the face region as a separate image
                (x1, y1, w1, h1) = cv2.boundingRect(np.array([leftEye]))
            else:
                (x1, y1, w1, h1) = cv2.boundingRect(np.array([rightEye]))

            roi = img[y1:y1 + h1, x1:x1 + w1]
            roi = imutils.resize(roi, width=250, height=100, inter=cv2.INTER_CUBIC)
    else:
        roi = None
    return roi
