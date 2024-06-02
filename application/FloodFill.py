from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal

from application.custom_functions import apply_circular_mask

# negativ kepre mukodik (a mutatok feherek)
class FloodFill(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.img_out = None

        h, w = self.img.shape[:2]
        self.im_mask = np.zeros((h+2, w+2), np.uint8)

        self.center = h//2, w//2
        self.seed_pt = self.center

        self.signal = UpdatedImageSignal()

        self.slider_spa = self.create_slider("Seed Point area (circle radius)", self.slider_value_change, 3, 30, 6)
        # self.slider_lo = self.create_slider("Lower treshold", self.slider_value_change, 0, 255, 250)
        # self.slider_up = self.create_slider("Upper treshold", self.slider_value_change, 0, 255, 255)

        layout = QVBoxLayout()
        layout.addWidget(self.slider_spa)
        # layout.addWidget(self.slider_lo)
        # layout.addWidget(self.slider_up)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.update_image()

    def create_slider(self, title, callback, min=0, max=100, init_value=0):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(init_value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
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
        radius = self.slider_spa.layout().itemAt(1).widget().value()
        # self.newSeedPoint(radius)

        flooded = self.img.copy()
        circ_mask = apply_circular_mask(flooded.shape, self.center, radius) # kozepen feher kor, hogy a floodfill biztosan mukodjon
        flooded[circ_mask] = 255
        self.im_mask[:] = 0
        cv2.floodFill(flooded, self.im_mask, self.seed_pt, 255, 250, 255, cv2.FLOODFILL_FIXED_RANGE)
        self.img_out = np.where(self.im_mask[1:-1, 1:-1] != 0, 255, 0).astype('uint8')
        self.signal.image_updated.emit()

        # kep megjelenitese szinesben, seed_pt es kezdokor szines
        img_col = cv2.cvtColor(self.img_out, cv2.COLOR_GRAY2BGR)
        img_col[self.seed_pt[0]-2:self.seed_pt[0]+2, self.seed_pt[1]-2:self.seed_pt[1]+2] = (255, 0, 0)
        cv2.circle(img_col, self.center, radius, (0, 0, 255), 3)
        height, width, channel = img_col.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_col.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def getNewImage(self):
        return self.img_out

    # def newSeedPoint(self, radius):
    #     if self.img[self.seed_pt[0], self.seed_pt[1]] != 255: # kep kozepe fekete, nem ora
    #         indices = np.argwhere(self.img == 255)
    #         print(indices)
    #         distances_squared = np.sum((indices - self.center)**2, axis=1)
    #         print(distances_squared)
    #         indices_in_circle = indices[distances_squared <= radius**2]
    #         print(indices_in_circle)
    #         self.seed_pt = indices_in_circle[np.random.choice(indices_in_circle.shape[0])]

