import os
import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QInputDialog, QCheckBox)

# a kepfeldolgozast vegzo osztalyok importalasa
from application.AddNoiseAdditive import AddNoiseAdditive
from application.AddNoiseSaltPepper import AddNoiseSaltPepper
from application.BlurDefault import BlurDefault
from application.BlurMedian import BlurMedian
from application.BlurGaussian import BlurGaussian
from application.Thresholding import Thresholding
from application.MorphologicalProcessing import MorphologicalProcessing
from application.FloodFill import FloodFill
from application.HoughLines import HoughLines
from application.TellTime import TellTime

# egyeb sajat fuggvenyek importalasa
from application.custom_functions import edgetrim, apply_circular_mask


# ez az osztaly tartalmazza a gui elemeket mely egy kep feldolgozasahoz szuksegesek
class ColumnWidget(QWidget):
    def __init__(self, title: str):
        super().__init__()

        self.image = None
        self.second_hand = False
        self.title = title
        self.lines = None

        self.initUI()

    def initUI(self):
        self.main_title_label = QLabel(self.title)
        self.main_title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        # Gombok kepek betoltesehez
        top_row = QHBoxLayout()

        select_image_btn = QPushButton("Select Image")
        select_image_btn.clicked.connect(self.selectImage)

        load_image_btn = QPushButton("Load New Image")
        load_image_btn.clicked.connect(self.loadImage)

        second_hand_chbx = QCheckBox("Second Hand")
        second_hand_chbx.stateChanged.connect(self.secondHandChanged)

        top_row.addWidget(select_image_btn)
        top_row.addWidget(load_image_btn)
        top_row.addWidget(second_hand_chbx)

        top_row_widget = QWidget()
        top_row_widget.setLayout(top_row)

        # Gombok szerkesztesi muveletek elvegzesere
        # Zaj hozzaadas
        apply_noise_row = QHBoxLayout()
        title_label = QLabel("Add noise")
        apply_noise_row.addWidget(title_label)

        gaussian_noise_btn = QPushButton("Apply Gaussian Noise")
        gaussian_noise_btn.clicked.connect(self.applyGaussianNoise)

        salt_pepper_noise_btn = QPushButton("Apply Salt & Pepper Noise")
        salt_pepper_noise_btn.clicked.connect(self.applySaltPepperNoise)

        apply_noise_row.addWidget(gaussian_noise_btn)
        apply_noise_row.addWidget(salt_pepper_noise_btn)

        apply_noise_row_widget = QWidget()
        apply_noise_row_widget.setLayout(apply_noise_row)

        # Simitasok zaj szuresere
        blur_row = QHBoxLayout()
        title_label = QLabel("Apply Blur")
        blur_row.addWidget(title_label)

        default_blur_btn = QPushButton("Default Blur")
        default_blur_btn.clicked.connect(self.applyDefaultBlur)

        median_blur_btn = QPushButton("Median Blur")
        median_blur_btn.clicked.connect(self.applyMedianBlur)

        gaussian_blur_btn = QPushButton("Gaussian Blur")
        gaussian_blur_btn.clicked.connect(self.applyGaussianBlur)

        blur_row.addWidget(default_blur_btn)
        blur_row.addWidget(median_blur_btn)
        blur_row.addWidget(gaussian_blur_btn)

        blur_row_widget = QWidget()
        blur_row_widget.setLayout(blur_row)

        # Kuszoboles
        treshold_row = QHBoxLayout()

        title_label = QLabel(f"Apply Treshold (Deafult: Otsu)")
        treshold_row.addWidget(title_label)

        treshold_btn = QPushButton("Apply Global Treshold")
        treshold_btn.clicked.connect(self.applyGlobalTreshold)

        inverse_btn = QPushButton("Inverse Image")
        inverse_btn.clicked.connect(self.inverseIm)

        treshold_row.addWidget(treshold_btn)
        treshold_row.addWidget(inverse_btn)

        treshold_row_widget = QWidget()
        treshold_row_widget.setLayout(treshold_row)

        # Erozio es Dilatacio
        processing_row = QHBoxLayout()

        title_label = QLabel("Processing steps")
        processing_row.addWidget(title_label)

        morph_btn = QPushButton("Apply Morphological Processing")
        morph_btn.clicked.connect(self.applyMorphologicalProcessing)

        floodfill_btn = QPushButton("Flood Fill")
        floodfill_btn.clicked.connect(self.applyFloodFill)

        hough_lines_btn = QPushButton("Apply Hough Lines")
        hough_lines_btn.clicked.connect(self.applyHoughLines)
        
        processing_row.addWidget(morph_btn)
        processing_row.addWidget(floodfill_btn)
        processing_row.addWidget(hough_lines_btn)

        processing_row_widget = QWidget()
        processing_row_widget.setLayout(processing_row)

        # Vonalkereses
        last_row = QHBoxLayout()

        title_label = QLabel("Last step: ")
        last_row.addWidget(title_label)
        
        tell_time_btn = QPushButton("TELL TIME!")
        tell_time_btn.clicked.connect(self.tellTime)
        tell_time_btn.setStyleSheet("""
            background-color: red;
            color: white;
            font-weight: bold;
            font-size: 16px;
        """)
        last_row.addWidget(tell_time_btn)

        last_row_widget = QWidget()
        last_row_widget.setLayout(last_row)

        # kep
        self.image_display = QLabel()

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.main_title_label)
        self.layout.addWidget(top_row_widget)
        self.layout.addWidget(apply_noise_row_widget)
        self.layout.addWidget(blur_row_widget)
        self.layout.addWidget(treshold_row_widget)
        self.layout.addWidget(processing_row_widget)
        self.layout.addWidget(last_row_widget)
        self.layout.addWidget(self.image_display)

        self.setLayout(self.layout)

    # teszteleshez hasznalt kepek betoltesere
    def selectImage(self):
        image_options = [f'img/faliora{i}.jpg' for i in range(15)]

        item, ok = QInputDialog.getItem(self, "Kep valasztasa", "Valasszon egy kepet:", image_options, 0, False)

        if ok and item:
            filename = item
        else:
            filename = 'img/faliora0.jpg'

        if filename:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.image = img
            self.title = filename

            new_display_widget = QLabel()
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image_display = new_display_widget
            self.displayImage(self.image, self.image_display)

    # egyeb kep hozzadasara
    def loadImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Kep feltoltese", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if filename:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # elofeldolgozas, ezek a lepesek el lettek vegezve a csatolt tesztkepeken is
            top, right, bottom, left = edgetrim(img) # egyszinu hatter levagasa a kep szelen
            img = img[top:bottom, left:right]
            img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA) # uj egyseges meret
            mask = apply_circular_mask(img.shape, (512//2, 512//2), 250) # az oran kivuli reszek kifeheritese
            img[~mask] = [255, 255, 255]
            self.image = img
            self.title = filename

            new_display_widget = QLabel()
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image_display = new_display_widget
            self.displayImage(self.image, self.image_display)

    def secondHandChanged(self, state):
        self.second_hand = bool(state)
        # if self.second_hand:
        #     print("Second hand checked")
        # else:
        #     print("Second hand unchecked")

    def inverseIm(self):
        if not self.image is None:
            new_display_widget = QLabel()
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image_display = new_display_widget
            self.image = 255 - self.image
            self.displayImage(self.image, self.image_display)

    # Zaj hozzaadasa
    def applyGaussianNoise(self):
        if not self.image is None:
            new_display_widget = AddNoiseAdditive(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    def applySaltPepperNoise(self):
        if not self.image is None:
            new_display_widget = AddNoiseSaltPepper(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    # Simitasok alkalmazasa
    def applyDefaultBlur(self):
        if not self.image is None:
            new_display_widget = BlurDefault(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    def applyMedianBlur(self):
        if not self.image is None:
            new_display_widget = BlurMedian(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    def applyGaussianBlur(self):
        if not self.image is None:
            new_display_widget = BlurGaussian(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    # Kuszoboles
    def applyGlobalTreshold(self):
        if not self.image is None:
            new_display_widget = Thresholding(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image = new_display_widget.getNewImage() # hogy akkor is frissuljon, ha nincs allitva semmi
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    # Erozio es Dilatacio
    def applyMorphologicalProcessing(self):
        if not self.image is None:
            new_display_widget = MorphologicalProcessing(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    # Szegmentalas
    def applyFloodFill(self):
        if not self.image is None:
            new_display_widget = FloodFill(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image = new_display_widget.getNewImage() # hogy akkor is frissuljon, ha nincs allitva semmi
            new_display_widget.signal.image_updated.connect(self.updateImage)
            self.image_display = new_display_widget

    # Vonalkereses
    def applyHoughLines(self):
        if not self.image is None:
            new_display_widget = HoughLines(self.image)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.lines = new_display_widget.getLines()
            new_display_widget.signal.image_updated.connect(self.updateLines)
            self.image_display = new_display_widget

    # Ido megallapitasa
    def tellTime(self):
        if not self.image is None or not self.lines is None:
            new_display_widget = TellTime(self.image, self.lines, self.second_hand)
            self.image_display.hide()
            self.layout.replaceWidget(self.image_display, new_display_widget)
            self.image_display = new_display_widget

    def updateLines(self):
        self.lines = self.image_display.getLines()

    def updateImage(self):
        self.image = self.image_display.getNewImage() # a metodus csak olyan metodusbol van hivva, ahol image_display egy olyan osztaly, ami biztosan tartalmazza a getNewImage metodust

    # kep megjelenitese, csak fekete fehert tud (2d array)
    def displayImage(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)