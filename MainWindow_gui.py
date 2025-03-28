# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(919, 638)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgLabel.sizePolicy().hasHeightForWidth())
        self.imgLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.imgLabel.setFont(font)
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imgLabel.setObjectName("imgLabel")
        self.horizontalLayout.addWidget(self.imgLabel)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioButton_left = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_left.setChecked(True)
        self.radioButton_left.setObjectName("radioButton_left")
        self.verticalLayout.addWidget(self.radioButton_left)
        self.radioButton_right = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_right.setObjectName("radioButton_right")
        self.verticalLayout.addWidget(self.radioButton_right)
        self.CreateDatasetbutton = QtWidgets.QPushButton(self.centralwidget)
        self.CreateDatasetbutton.setObjectName("CreateDatasetbutton")
        self.verticalLayout.addWidget(self.CreateDatasetbutton)
        self.Trainingbutton = QtWidgets.QPushButton(self.centralwidget)
        self.Trainingbutton.setObjectName("Trainingbutton")
        self.verticalLayout.addWidget(self.Trainingbutton)
        self.openCameraButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openCameraButton.sizePolicy().hasHeightForWidth())
        self.openCameraButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.openCameraButton.setFont(font)
        self.openCameraButton.setObjectName("openCameraButton")
        self.verticalLayout.addWidget(self.openCameraButton)
        self.stopCameraButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.stopCameraButton.setFont(font)
        self.stopCameraButton.setObjectName("stopCameraButton")
        self.verticalLayout.addWidget(self.stopCameraButton)
        self.startDetectionButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.startDetectionButton.sizePolicy().hasHeightForWidth())
        self.startDetectionButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.startDetectionButton.setFont(font)
        self.startDetectionButton.setObjectName("startDetectionButton")
        self.verticalLayout.addWidget(self.startDetectionButton)
        self.stopDetectionButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.stopDetectionButton.setFont(font)
        self.stopDetectionButton.setObjectName("stopDetectionButton")
        self.verticalLayout.addWidget(self.stopDetectionButton)
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exitButton.sizePolicy().hasHeightForWidth())
        self.exitButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.verticalLayout.addWidget(self.exitButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.detected_directions_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setUnderline(False)
        self.detected_directions_text.setFont(font)
        self.detected_directions_text.setFrameShape(QtWidgets.QFrame.Box)
        self.detected_directions_text.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.detected_directions_text.setText("")
        self.detected_directions_text.setAlignment(QtCore.Qt.AlignCenter)
        self.detected_directions_text.setObjectName("detected_directions_text")
        self.verticalLayout.addWidget(self.detected_directions_text)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.imgLabel.setText(_translate("MainWindow", "Live Camera Feed"))
        self.radioButton_left.setText(_translate("MainWindow", "Left Eye"))
        self.radioButton_right.setText(_translate("MainWindow", "Right Eye"))
        self.CreateDatasetbutton.setText(_translate("MainWindow", "Create Dataset"))
        self.Trainingbutton.setText(_translate("MainWindow", "Training"))
        self.openCameraButton.setText(_translate("MainWindow", "Open Camera"))
        self.stopCameraButton.setText(_translate("MainWindow", "Stop Camera"))
        self.startDetectionButton.setText(_translate("MainWindow", "Start Detection"))
        self.stopDetectionButton.setText(_translate("MainWindow", "Stop Detection"))
        self.exitButton.setText(_translate("MainWindow", "Exit"))
        self.label.setText(_translate("MainWindow", "Detected Directions"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

