import cv2
import sys
import os
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import MainWindow_gui
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
import winsound
import win32com.client as wincl
import warnings
import time
import serial


class SerialPortDataProcessThread(QThread):

    def __init__(self, serialPort):
        super(SerialPortDataProcessThread, self).__init__()
        self.serialPort = serialPort

    def run(self):
        self.exec_()

    def Close_Serial(self):
        self.serialPort.close()

    @pyqtSlot(object)
    def writeData(self, data):
        data = str(data)
        data_encoded = data.encode("utf-8")
        self.serialPort.write(data_encoded)


# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #

class WheelChair_Class(QMainWindow, MainWindow_gui.Ui_MainWindow):
    serialWrite = QtCore.pyqtSignal(object)

    def __init__(self):
        super(WheelChair_Class, self).__init__()

        # self.setupUi(self)

        loadUi("MainWindow_gui.ui", self)

        self.SERIAL_PORT_ENABLE = False
        self.left_counter = 0
        self.right_counter = 0
        self.up_counter = 0
        self.down_counter = 0
        self.stop_counter = 0
        self.COUNTER_THRESHOLD = 1

        if self.SERIAL_PORT_ENABLE:
            self.comm_port = "COM4"

            self.serialConnection = serial.Serial(port=self.comm_port, baudrate=9600)
            self.SerialThread = SerialPortDataProcessThread(self.serialConnection)
            self.serialWrite.connect(self.SerialThread.writeData)

        self.CreateDatasetbutton.clicked.connect(self.CreateDatasetClicked)
        self.Trainingbutton.clicked.connect(self.TrainingClicked)
        self.openCameraButton.clicked.connect(self.openCameraClicked)
        self.stopCameraButton.clicked.connect(self.stopCameraClicked)
        self.startDetectionButton.clicked.connect(self.startAllDetection)
        self.stopDetectionButton.clicked.connect(self.stopAllDetection)
        self.exitButton.clicked.connect(self.exitClicked)

        print("[INFO] loading facial landmark predictor...")
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        names_file = Path("./train_model_data.pkl")

        if names_file.is_file():
            with open('train_model_data.pkl', 'rb') as f:
                [self.model, self.eyeFeatures, self.dir_ids, self.dir_labels] = pickle.load(f)
        else:
            print('No training data File exists')

        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 20
        self.COUNTER = 0
        self.YAWN_COUNTER = 0
        self.FACE_COUNTER = 0
        self.ALARM_ON = False
        self.start_detection_Flag = False
        self.frequency = 2500
        self.duration = 500
        self.speak = wincl.Dispatch("SAPI.SpVoice")
        self.systemStatusFlag = False

    def sendData(self, buff):
        self.serialWrite.emit(buff)

    @pyqtSlot(bool)
    def updateSystemStatus(self, status):
        self.systemStatusFlag = status

    @pyqtSlot()
    def startAllDetection(self):
        self.start_detection_Flag = True

    @pyqtSlot()
    def stopAllDetection(self):
        self.start_detection_Flag = False

    @pyqtSlot()
    def CreateDatasetClicked(self):

        if self.radioButton_left.isChecked():
            left_or_right_eye = "left_eye"
        else:
            left_or_right_eye = "right_eye"

        dataset_path = "dataset"

        directory_left = "1.left"
        directory_right = "2.right"
        directory_up = "3.up"
        directory_down = "4.down"
        directory_stop = "5.stop"

        path_left = os.path.join(dataset_path, directory_left)
        path_right = os.path.join(dataset_path, directory_right)
        path_up = os.path.join(dataset_path, directory_up)
        path_down = os.path.join(dataset_path, directory_down)
        path_stop = os.path.join(dataset_path, directory_stop)

        try:
            shutil.rmtree(path_left)
            shutil.rmtree(path_right)
            shutil.rmtree(path_up)
            shutil.rmtree(path_down)
            shutil.rmtree(path_stop)
            print("Directory '%s' has been removed successfully" % directory_left)
            print("Directory '%s' has been removed successfully" % directory_right)
            print("Directory '%s' has been removed successfully" % directory_up)
            print("Directory '%s' has been removed successfully" % directory_down)
            print("Directory '%s' has been removed successfully" % directory_stop)
        except OSError as error:
            print(error)
            print("Directory '%s' can not be removed" % directory_left)
            print("Directory '%s' can not be removed" % directory_right)
            print("Directory '%s' can not be removed" % directory_up)
            print("Directory '%s' can not be removed" % directory_down)
            print("Directory '%s' can not be removed" % directory_stop)

        os.mkdir(path_left)
        os.mkdir(path_right)
        os.mkdir(path_up)
        os.mkdir(path_down)
        os.mkdir(path_stop)

        self.openCameraClicked()

        self.showMessagebox("Create Dataset for LEFT Direction")

        for i in range(100):
            eye_sample = self.EyeSampler(self.image, left_or_right_eye)
            if eye_sample is not None:
                filename = path_left + "/left__" + str(i) + '.jpeg'
                cv2.imwrite(filename, eye_sample)

        self.showMessagebox("Create Dataset for RIGHT Direction")

        for i in range(100):
            eye_sample = self.EyeSampler(self.image, left_or_right_eye)
            if eye_sample is not None:
                filename = path_right + "/right__" + str(i) + '.jpeg'
                cv2.imwrite(filename, eye_sample)

        self.showMessagebox("Create Dataset for UP Direction")

        for i in range(100):
            eye_sample = self.EyeSampler(self.image, left_or_right_eye)
            if eye_sample is not None:
                filename = path_up + "/up__" + str(i) + '.jpeg'
                cv2.imwrite(filename, eye_sample)

        self.showMessagebox("Create Dataset for DOWN Direction")

        for i in range(100):
            eye_sample = self.EyeSampler(self.image, left_or_right_eye)
            if eye_sample is not None:
                filename = path_down + "/down__" + str(i) + '.jpeg'
                cv2.imwrite(filename, eye_sample)

        self.showMessagebox("Create Dataset for STOP Direction")

        for i in range(100):
            eye_sample = self.EyeSampler(self.image, left_or_right_eye)
            if eye_sample is not None:
                filename = path_stop + "/stop__" + str(i) + '.jpeg'
                cv2.imwrite(filename, eye_sample)

        self.showMessagebox("DONE !")

    @pyqtSlot()
    def TrainingClicked(self):

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

        self.showMessagebox("TRAINING DONE !")

    def showMessagebox(self, text):
        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        # mb.setText("your text")
        mb.setText(text)
        mb.setWindowTitle("Warning")
        mb.setStandardButtons(QMessageBox.Ok)
        mb.exec_()

    def EyeSampler(self, img, left_or_right_eye):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # print(rects)

        roi = None

        if len(rects):

            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]

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

    def eye_direction_detector(self, img):

        if self.radioButton_left.isChecked():
            left_or_right_eye = "left_eye"
        else:
            left_or_right_eye = "right_eye"

        eye_sample = self.EyeSampler(img, left_or_right_eye)

        if eye_sample is not None:

            gray = cv2.cvtColor(eye_sample, cv2.COLOR_BGR2GRAY)
            logo = cv2.resize(gray, (250, 100))

            (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)

            pred = self.model.predict(H.reshape(1, -1))[0]
            print(pred, self.dir_labels[pred - 1])

            if self.SERIAL_PORT_ENABLE:
                self.sendData(pred)

            predicted_label = self.dir_labels[pred - 1]
            print(predicted_label)
            self.detected_directions_text.setText(predicted_label)

            if predicted_label == "left":
                self.left_counter += 1
                if self.left_counter > self.COUNTER_THRESHOLD:
                    self.detected_directions_label.setPixmap(QPixmap('./images/arrow-left.png'))
                    self.left_counter = 0
                    self.right_counter = 0
                    self.up_counter = 0
                    self.down_counter = 0
                    self.stop_counter = 0

            elif predicted_label == "right":
                self.right_counter += 1
                if self.right_counter > self.COUNTER_THRESHOLD:
                    self.detected_directions_label.setPixmap(QPixmap('./images/right-arrow.png'))
                    self.left_counter = 0
                    self.right_counter = 0
                    self.up_counter = 0
                    self.down_counter = 0
                    self.stop_counter = 0

            elif predicted_label == "up":
                self.up_counter += 1
                if self.up_counter > self.COUNTER_THRESHOLD:
                    self.detected_directions_label.setPixmap(QPixmap('./images/top-up.png'))
                    self.left_counter = 0
                    self.right_counter = 0
                    self.up_counter = 0
                    self.down_counter = 0
                    self.stop_counter = 0

            elif predicted_label == "down":
                self.down_counter += 1
                if self.down_counter > self.COUNTER_THRESHOLD:
                    self.detected_directions_label.setPixmap(QPixmap('./images/down.png'))
                    self.left_counter = 0
                    self.right_counter = 0
                    self.up_counter = 0
                    self.down_counter = 0
                    self.stop_counter = 0

            elif predicted_label == "stop":
                self.stop_counter += 1
                if self.stop_counter > self.COUNTER_THRESHOLD:
                    self.detected_directions_label.setPixmap(QPixmap('./images/stop.png'))
                    self.left_counter = 0
                    self.right_counter = 0
                    self.up_counter = 0
                    self.down_counter = 0
                    self.stop_counter = 0

            cv2.putText(img, self.dir_labels[pred - 1], (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.DisplayImage(img)

    @pyqtSlot()
    def openCameraClicked(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0.1)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.start_detection_Flag:
            # self.blinkDetector(self.image)
            self.eye_direction_detector(self.image)
        else:
            self.DisplayImage(self.image, 1)

    @pyqtSlot()
    def stopCameraClicked(self):
        self.timer.stop()
        self.capture.release()

    def DisplayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

    @pyqtSlot()
    def exitClicked(self):
        if self.SERIAL_PORT_ENABLE:
            self.SerialThread.Close_Serial()
        QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WheelChair_Class()
    window.show()
    app.exec_()
