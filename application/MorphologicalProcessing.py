from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout, QComboBox, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal

class MorphologicalProcessing(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.img_out = img.copy()
        self.structelem = "rect"
        self.kernel_size = 5
        self.signal = UpdatedImageSignal()

        # StructuringElement tipusanak meghatarozasa, illetve egy db erozio v. dilatacio lepes elvegzese
        self.options_row = QHBoxLayout()

        self.structelem_dropdown = QComboBox()
        self.structelem_dropdown.addItem("Rectangle", "rect")
        self.structelem_dropdown.addItem("Ellipse", "ellipse")
        self.structelem_dropdown.addItem("Cross", "cross")
        self.structelem_dropdown.currentIndexChanged.connect(self.setStructElem)

        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(self.update_by_dilate)
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.clicked.connect(self.update_by_erode)

        self.options_row.addWidget(self.structelem_dropdown)
        self.options_row.addWidget(self.dilate_btn)
        self.options_row.addWidget(self.erode_btn)

        self.options_row_widget = QWidget()
        self.options_row_widget.setLayout(self.options_row)
        
        self.slider = self.create_slider("Kernel size", self.slider_value_change)

        layout = QVBoxLayout()
        layout.addWidget(self.options_row_widget)
        layout.addWidget(self.slider)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # self.update_image()
        height, width = self.img.shape
        bytesPerLine = width
        qImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def create_slider(self, title, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(3)
        slider.setValue(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(callback)

        value_label = QLabel()
        value_label.setText(title + ": " + str(slider.value()*2+3))

        slider.valueChanged.connect(lambda value: value_label.setText(title + ": " + str(value*2+3)))

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(value_label)
        slider_layout.addWidget(slider)

        slider_widget = QWidget()
        slider_widget.setLayout(slider_layout)

        return slider_widget

    def slider_value_change(self, value):
        self.kernel_size = value*2+3

    def update_by_dilate(self):
        form = cv2.MORPH_RECT
        if self.structelem == "ellipse":
            form = cv2.MORPH_ELLIPSE
        elif self.structelem == "cross":
            form = cv2.MORPH_CROSS
        
        kernel = cv2.getStructuringElement(form, (self.kernel_size, self.kernel_size))

        self.img_out = cv2.dilate(self.img_out, kernel)

        self.signal.image_updated.emit()

        height, width = self.img_out.shape
        bytesPerLine = width
        qImg = QImage(self.img_out.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    def update_by_erode(self):
        form = cv2.MORPH_RECT
        if self.structelem == "ellipse":
            form = cv2.MORPH_ELLIPSE
        elif self.structelem == "cross":
            form = cv2.MORPH_CROSS
        
        kernel = cv2.getStructuringElement(form, (self.kernel_size, self.kernel_size))

        self.img_out = cv2.erode(self.img_out, kernel)

        self.signal.image_updated.emit()

        height, width = self.img_out.shape
        bytesPerLine = width
        qImg = QImage(self.img_out.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    def getNewImage(self):
        return self.img_out

    def setStructElem(self, index):
        self.structelem = self.structelem_dropdown.itemData(index)

