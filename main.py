import os
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QInputDialog)

from application.ColumnWidget import ColumnWidget


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("What's the time?")
        # self.main_widget = QWidget(self)
        # self.setCentralWidget(self.main_widget)

        # Up to 3 images can be loaded at the same time, so processing steps can be compared nicely
        self.image1 = ColumnWidget("Image 1")
        self.image2 = ColumnWidget("Image 2")
        self.image3 = ColumnWidget("Image 3")

        layout = QHBoxLayout() #self.main_widget)
        layout.addWidget(self.image1)
        layout.addWidget(self.image2)
        layout.addWidget(self.image3)

        self.setLayout(layout)
             


if __name__ == '__main__':
    app = QApplication([])
    window = ImageProcessingApp()
    window.show()
    app.exec()
