from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal

class Thresholding(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.otsu_threshold, self.img_out = cv2.threshold(self.img, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.signal = UpdatedImageSignal()

        title = f"Threshold (Otsu: {self.otsu_threshold})"
        self.slider = self.create_slider(title, self.slider_value_change)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.update_image()

    def create_slider(self, title, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(255)
        slider.setValue(self.otsu_threshold)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(25)
        slider.valueChanged.connect(callback)

        value_label = QLabel()
        value_label.setText(title + ": " + str(slider.value()))

        slider.valueChanged.connect(lambda value: value_label.setText(title + ": " + str(value)))

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(value_label)
        slider_layout.addWidget(slider)

        slider_widget = QWidget()
        slider_widget.setLayout(slider_layout)

        return slider_widget

    def slider_value_change(self, value):
        self.update_image()

    def update_image(self):
        pos = self.slider.layout().itemAt(1).widget().value()

        threshold, self.img_out = cv2.threshold(self.img, pos, 255, cv2.THRESH_BINARY)

        self.signal.image_updated.emit()

        height, width = self.img_out.shape
        bytesPerLine = width
        qImg = QImage(self.img_out.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    def getNewImage(self):
        return self.img_out

