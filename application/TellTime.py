from PySide6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import json

from application.UpdatedImageSignal import UpdatedImageSignal

from application.TellTimeDataDashboard import ResultDashboardWindow
from application.tell_time_helpers import process_data, tell_time, closest_line_mask

# az ido megmondasa a vonalakbol illetve a szegmentalt mutato kepbol
class TellTime(QWidget):
    def __init__(self, img, lines, second_hand):
        super().__init__()

        self.image_label = QLabel()
        self.db_window = None
        self.img = img.copy()
        self.img_out = None
        self.lines = lines
        self.second_hand = second_hand

        # hough vonalak feldolgozasa
        self.clock_dict, self.labelled_lines_dict, self.cl_centers = process_data(self.img, self.lines, self.second_hand)
        self.time_str = tell_time(self.clock_dict, self.second_hand)
        print('THE TIME IS:  ' + self.time_str)

        self.time_label = QLabel()
        self.time_label.setText('THE TIME IS: ' + self.time_str)
        self.time_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        # self.signal = UpdatedImageSignal()

        layout = QVBoxLayout()
        layout.addWidget(self.time_label)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.create_data_dashbord()

        self.update_image()

    def create_data_dashbord(self):
        self.db_window = ResultDashboardWindow(self.cl_centers, self.labelled_lines_dict, self.clock_dict)
        self.db_window.show()

    def update_image(self):
        # kep megjelenitese szinesben, ora: piros, perc: zold, masodperc: kek
        hour_line = self.clock_dict['hour']['line']
        minute_line = self.clock_dict['minute']['line']


        if self.clock_dict['second'] is not None:
            second_line = self.clock_dict['second']['line']
            hour_mask = closest_line_mask(self.img, hour_line, [minute_line, second_line])
            minute_mask = closest_line_mask(self.img, minute_line, [hour_line, second_line])
            second_mask = closest_line_mask(self.img, second_line, [hour_line, minute_line])
        else:
            hour_mask = closest_line_mask(self.img, hour_line, [minute_line])
            minute_mask = closest_line_mask(self.img, minute_line, [hour_line])
        
        img_col = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        b, g, r = cv2.split(img_col)
        if hour_line[0] != minute_line[0] and hour_line[1] != minute_line[1]:
            b[hour_mask] = 0
            g[hour_mask] = 0
            b[minute_mask] = 0
            r[minute_mask] = 0
            if self.clock_dict['second'] is not None:
                g[second_mask] = 0
                r[second_mask] = 0
        img_col = cv2.merge((b, g, r))

        height, width, channel = img_col.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_col.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qImg))



