from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal

class BlurMedian(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.img_out = None
        self.signal = UpdatedImageSignal()

        self.slider = self.create_slider("Kernel size", self.slider_value_change)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.update_image()

    def create_slider(self, title, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(20)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
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
        self.update_image()

    def update_image(self):
        kernel_size = self.slider.layout().itemAt(1).widget().value() * 2 + 3

        self.img_out = cv2.medianBlur(self.img, kernel_size)

        self.signal.image_updated.emit()

        height, width = self.img_out.shape
        bytesPerLine = width
        qImg = QImage(self.img_out.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    def getNewImage(self):
        return self.img_out


