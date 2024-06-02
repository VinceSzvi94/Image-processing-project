from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout, QComboBox, QPushButton, QCheckBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import math
import cv2

from application.UpdatedImageSignal import UpdatedImageSignal


def draw_lines(img, lines):
    sizemax = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Computing line endpoints outside of image matrix
            pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
            pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
    
    return img


# negativ kepre mukodik (a mutatok feherek), a kep nem frissul csak a vonalak
# lehetoseg van a mutatok vekonyitasara
class HoughLines(QWidget):
    def __init__(self, img):
        super().__init__()

        self.image_label = QLabel()
        self.img = img.copy()
        self.img_out = None
        self.img_thinned = None
        self.lines = None
        self.thinned = False
        self.th_type = "zs"
        self.structelem = "rect"
        self.kernel_size = 3

        self.signal = UpdatedImageSignal()

        # row for thining
        self.thin_row = QHBoxLayout()

        is_thinned_chbx = QCheckBox("Thinning applied")
        is_thinned_chbx.stateChanged.connect(self.isThinnedChanged)

        self.type_dropdown = QComboBox()
        self.type_dropdown.addItem("Zhang-Suen", "zs")
        self.type_dropdown.addItem("Guo-Hall", "gh")
        self.type_dropdown.currentIndexChanged.connect(self.setThinningType)

        self.thin_row.addWidget(is_thinned_chbx)
        self.thin_row.addWidget(self.type_dropdown)

        self.thin_row_widget = QWidget()
        self.thin_row_widget.setLayout(self.thin_row)

        # row for dilating thinned img
        self.dilate_row = QHBoxLayout()

        self.structelem_dropdown = QComboBox()
        self.structelem_dropdown.addItem("Rectangle", "rect")
        self.structelem_dropdown.addItem("Ellipse", "ellipse")
        self.structelem_dropdown.addItem("Cross", "cross")
        self.structelem_dropdown.currentIndexChanged.connect(self.setStructElem)

        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(self.update_by_dilate)

        self.dilate_row.addWidget(self.structelem_dropdown)
        self.dilate_row.addWidget(self.dilate_btn)

        self.dilate_row_widget = QWidget()
        self.dilate_row_widget.setLayout(self.dilate_row)

        # slider for hough line threshold
        self.slider_th = self.create_slider("Treshhold", self.slider_value_change, 20, 450, 200)

        layout = QVBoxLayout()
        layout.addWidget(self.thin_row_widget)
        layout.addWidget(self.dilate_row_widget)
        layout.addWidget(self.slider_th)
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
        th = self.slider_th.layout().itemAt(1).widget().value()
        if self.thinned:
            self.lines = cv2.HoughLines(self.img_thinned, 1, np.pi/180, th, None, 0, 0)
        else:
            self.lines = cv2.HoughLines(self.img, 1, np.pi/180, th, None, 0, 0)
        self.signal.image_updated.emit()

        # kep megjelenitese szinesben, vonalak sargak, thinned kep piros
        img_col = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        img_col = draw_lines(img_col, self.lines)
        if self.img_thinned is not None:
            b, g, r = cv2.split(img_col)
            b = cv2.bitwise_and(b, ~self.img_thinned)
            g = cv2.bitwise_and(g, ~self.img_thinned)
            img_col = cv2.merge((b, g, r))

        height, width, channel = img_col.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_col.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    def update_by_thinning(self):
        if self.th_type == "zs":
            self.img_thinned = cv2.ximgproc.thinning(self.img, None, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        elif self.th_type == "gh":
            self.img_thinned = cv2.ximgproc.thinning(self.img, None, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        else:
            self.img_thinned = None
        
        self.update_image()
    
    def update_by_dilate(self):
        form = cv2.MORPH_RECT
        if self.structelem == "ellipse":
            form = cv2.MORPH_ELLIPSE
        elif self.structelem == "cross":
            form = cv2.MORPH_CROSS
        
        kernel = cv2.getStructuringElement(form, (self.kernel_size, self.kernel_size))

        if self.img_thinned is None:
            print("No thinned image")
        else:
            self.img_thinned = cv2.dilate(self.img_thinned, kernel)

        self.update_image()

    def getNewImage(self):
        return self.img_out
    
    def getLines(self):
        return self.lines
    
    def setStructElem(self, index):
        self.structelem = self.structelem_dropdown.itemData(index)
    
    def isThinnedChanged(self, state):
        self.thinned = bool(state)
        if self.thinned:
            self.update_by_thinning()
        else:
            self.img_thinned = None
            self.update_image()
    
    def setThinningType(self, index):
        self.th_type = self.type_dropdown.itemData(index)
        self.update_by_thinning()


