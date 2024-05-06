import os
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QInputDialog)


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image = None
        # {
        #     'custom': None,
        #     # felhasznalt kepek atalakitasa: 420x420 meretre hozas + egyszinu szegelyek levagasa, lasd: own_functions -> edgetrim
        #     'faliora0': cv2.imread('img/faliora0.jpg'),
        #     'faliora1': cv2.imread('img/faliora1.jpg'),
        #     'faliora2': cv2.imread('img/faliora2.jpg'),
        #     'faliora3': cv2.imread('img/faliora3.jpg'),
        #     'faliora4': cv2.imread('img/faliora4.jpg'),
        #     'faliora5': cv2.imread('img/faliora5.jpg'),
        #     'faliora6': cv2.imread('img/faliora6.jpg'),
        #     'faliora7': cv2.imread('img/faliora7.jpg'),
        #     'faliora8': cv2.imread('img/faliora8.jpg'),
        #     'faliora9': cv2.imread('img/faliora9.jpg'),
        #     'faliora10': cv2.imread('img/faliora10.jpg'),
        #     'faliora11': cv2.imread('img/faliora11.jpg'),
        #     'faliora12': cv2.imread('img/faliora12.jpg'),
        #     'faliora13': cv2.imread('img/faliora13.jpg'),
        #     'faliora14': cv2.imread('img/faliora14.jpg'),
        #     'faliora15': cv2.imread('img/faliora15.jpg'),
        # }
        # feladat megoldasa soran felhasznalt kepek betoltese
        # for i in range(16):
        #     self.images[f'faliora{i}'] = cv2.resize(self.images[f'faliora{i}'], (420, 420))
        self.processing_steps = []

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Processing App")

        # Creating layout

        # Load Image button
        load_image_btn = QPushButton("Load Image")
        load_image_btn.clicked.connect(self.loadImage)

        # Save Processed Image button
        save_image_btn = QPushButton("Save Processed Image")
        save_image_btn.clicked.connect(self.saveProcessedImage)

        # Process Image button
        process_image_btn = QPushButton("Process Image")
        process_image_btn.clicked.connect(self.processImage)

        # Display widgets
        self.original_image_label = QLabel()
        self.processed_image_label = QLabel()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(load_image_btn)
        layout.addWidget(self.original_image_label)
        layout.addWidget(process_image_btn)
        layout.addWidget(self.processed_image_label)
        layout.addWidget(save_image_btn)

        self.setLayout(layout)

    def loadImage(self):
        image_options = [f'img/faliora{i}.jpg' for i in range(16)]

        item, ok = QInputDialog.getItem(self, "Select Image", "Choose an image:", image_options, 0, False)

        if ok and item:
            # If the user selected an option from the list, load that image
            print(f'item as filename-> {item} was ok: {ok}')
            filename = item
        else:
            # If the user did not select an option from the list, use QFileDialog to select a custom image
            print(f'selecting custom imgage, was ok: {ok}')
            filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            self.image = cv2.imread(filename)
            self.displayImage(self.image, self.original_image_label)
        

    def processImage(self):
        if self.image is not None:
            # Perform image processing here
            self.processed_image = self.image  # Placeholder for actual image processing

            self.displayImage(self.processed_image, self.processed_image_label)

    def saveProcessedImage(self):
        if self.processed_image is not None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if filename:
                cv2.imwrite(filename, self.processed_image)

    def displayImage(self, image, label):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication([])
    window = ImageProcessingApp()
    window.show()
    app.exec()
