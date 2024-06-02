from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal

def add_point_noise(img_in, percentage, value):
    noise_res = np.copy(img_in)
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)

    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])

        if img_in.ndim == 2:
            noise_res[j, i] = value

        if img_in.ndim == 3:
            noise_res[j, i] = [value, value, value]

    return noise_res

class AddNoiseSaltPepper(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.img_out = None
        self.signal = UpdatedImageSignal()

        self.slider_s = self.create_slider("Salt", self.slider_value_change)
        self.slider_p = self.create_slider("Pepper", self.slider_value_change)

        layout = QVBoxLayout()
        layout.addWidget(self.slider_s)
        layout.addWidget(self.slider_p)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.update_image()

    def create_slider(self, title, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(25)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
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
        salt = self.slider_s.layout().itemAt(1).widget().value()
        pepper = self.slider_p.layout().itemAt(1).widget().value()

        n1 = add_point_noise(self.img, salt/100, 255)
        self.img_out = add_point_noise(n1, pepper/100, 0)

        self.signal.image_updated.emit()

        height, width = self.img_out.shape
        bytesPerLine = width
        qImg = QImage(self.img_out.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def getNewImage(self):
        return self.img_out

