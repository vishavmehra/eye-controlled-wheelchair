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
#left_or_right_eye = "right_eye"

dataset_path = "dataset"

directory_left = "1.left"
directory_right = "2.right"

path_left = os.path.join(dataset_path, directory_left)
path_right = os.path.join(dataset_path, directory_right)


# try:
#     if os.path.exists(path_left):
#         shutil.rmtree(path_left)
#         print("Directory '%s' has been removed successfully" % directory_left)
#     if os.path.exists(path_right):
#         shutil.rmtree(path_right)
#         print("Directory '%s' has been removed successfully" % directory_right)
# except OSError as error:
#     print(error)
#     print("Directory '%s' can not be removed" % directory_left)
#     print("Directory '%s' can not be removed" % directory_right)

os.mkdir(path_left)
os.mkdir(path_right)


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

for i in range(100):
      
    # Capture the video frame 
    # by frame 
    ret, frame = capture.read() 
    
    if ret:
  
        # Display the resulting frame 
        frame = cv2.flip(frame, 1)      
        # cv2.imshow('frame', frame) 

        eye_sample = EyeSampler(frame, left_or_right_eye)
        
        if eye_sample.any():
            cv2.imshow('EYE', eye_sample) 
            
            if left_or_right_eye == "left_eye":
                filename = path_left + "/left__" + str(i) + '.jpeg'
            else:
                filename = path_right + "/right__" + str(i) + '.jpeg'
                
            cv2.imwrite(filename, eye_sample)
        
      
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
# After the loop release the cap object 
capture.release() 
# Destroy all the windows 
cv2.destroyAllWindows()