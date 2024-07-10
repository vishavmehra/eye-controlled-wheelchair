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

from functions import *

left_or_right_eye = "left_eye"
# left_or_right_eye = "right_eye"

with open('train_model_data.pkl', 'rb') as f:
    [model, eyeFeatures, dir_ids, dir_labels] = pickle.load(f)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)


while True:
    
    ret, frame = capture.read() 
    
    if ret:
        frame = cv2.flip(frame, 1)
    
        eye_sample = EyeSampler(frame, left_or_right_eye)
        
        if eye_sample is not None:
            gray = cv2.cvtColor(eye_sample, cv2.COLOR_BGR2GRAY)
            logo = cv2.resize(gray, (250, 100))

            (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)

            pred = model.predict(H.reshape(1, -1))[0]
            # print(pred, dir_labels[pred - 1])
            
            predicted_label = dir_labels[pred - 1]
            print(predicted_label)
            
            cv2.putText(frame, predicted_label, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('frame', frame) 
            
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
capture.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 



