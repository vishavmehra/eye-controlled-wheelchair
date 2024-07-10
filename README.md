# Wheelchair Control System

This repository contains code for a wheelchair control system using eye-tracking and facial landmarks. The system allows users to control the wheelchair by moving their eyes in different directions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Credits](#credits)

## Introduction
This project aims to provide a hands-free control system for wheelchair users by leveraging eye-tracking and facial landmark detection. The system uses a webcam to capture the user's face and eyes, processes the images to detect eye movements, and translates these movements into wheelchair control commands.

## Features
- Real-time eye movement detection
- Training mode to create a dataset of eye movements
- Five control commands: left, right, up, down, and stop
- Integration with a serial port to send control commands to the wheelchair

## Installation
### Prerequisites
- Python 3.7 or higher
- Anaconda
- Webcam

### Steps
1. Clone the repository:
   ```git clone https://github.com/vishavmehra/eye-controlled-wheelchair.git```
   ```cd eye-controlled-wheelchair```
2. Create and activate a new conda environment:
   ```conda create -n eye-controlled-wheelchair python=3.8```
   ```conda activate eye-controlled-wheelchair```
3. Install the required packages:
   ```pip install -r requirements.txt```
4. Download the pre-trained facial landmark predictor model:
   ```wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2```
  ```bunzip2 shape_predictor_68_face_landmarks.dat.bz2```

 ### Usage
 1. Run the main application:
   ```python main.py```
 2. Use the GUI to create a dataset, train the model, and start/stop the camera and detection processes.
 ### File Structure
    wheelchair-control-system/
```
├── dataset/                  # Directory for storing the dataset
├── images/                   # Directory for storing arrow images used in the GUI
├── MainWindow_gui.ui         # UI file for the main window
├── MainWindow_gui.py         # Python code generated from the UI file
├── main.py                   # Main entry point of the application
├── requirements.txt          # List of required packages
├── shape_predictor_68_face_landmarks.dat  # Pre-trained facial landmark predictor
└── README.md                 # This README file
```
### Dependencies
1. Python 3.7 or higher
2. OpenCV
3. PyQt5
4. Dlib
5. Imutils
6. Scikit-image
7. Scikit-learn
8. Winsound
9. PyWin32

### Credits
This project was developed using the following libraries and resources:
OpenCV
Dlib
PyQt5
Imutils
Scikit-image
Scikit-learn

It also contains multiple scripts for better understanding of the codeflow which was then incorporated into one class in the final algorithm.
Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or feedback, please contact mehravishav@gmail.com.
