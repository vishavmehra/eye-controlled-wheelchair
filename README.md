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
   ```sh
   git clone https://github.com/yourusername/wheelchair-control-system.git
   cd wheelchair-control-system
Create and activate a new conda environment:

sh
Copy code
conda create -n wheelchair_control python=3.8
conda activate wheelchair_control
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Download the pre-trained facial landmark predictor model:

sh
Copy code
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
Usage
Run the main application:

sh
Copy code
python main.py
Use the GUI to create a dataset, train the model, and start/stop the camera and detection processes.

File Structure
perl
Copy code
wheelchair-control-system/
│
├── dataset/                  # Directory for storing the dataset
├── images/                   # Directory for storing arrow images used in the GUI
├── MainWindow_gui.ui         # UI file for the main window
├── MainWindow_gui.py         # Python code generated from the UI file
├── main.py                   # Main entry point of the application
├── requirements.txt          # List of required packages
├── shape_predictor_68_face_landmarks.dat  # Pre-trained facial landmark predictor
└── README.md                 # This README file
Dependencies
Python 3.7 or higher
OpenCV
PyQt5
Dlib
Imutils
Scikit-image
Scikit-learn
Winsound
PyWin32
You can install the required packages using the provided requirements.txt file.

Credits
This project was developed using the following libraries and resources:

OpenCV
Dlib
PyQt5
Imutils
Scikit-image
Scikit-learn
Special thanks to the developers and contributors of these projects.
It also contains multiple scripts for better understanding of the codeflow which was then incorporated into one class in the final algorithm.
Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or feedback, please contact mehravishav@gmail.com.