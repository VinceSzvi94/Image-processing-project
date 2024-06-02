import numpy as np
import math
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


# plot az eredmenyekrol
class MyMplCanvas(FigureCanvasQTAgg):
    def __init__(self, centers, lines: dict, clock_dict: dict, parent=None):

        fig = Figure(figsize=(14, 11))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])
        self.axes1 = fig.add_subplot(gs[:, 0])
        self.axes2 = fig.add_subplot(gs[0, 1])
        self.axes3 = fig.add_subplot(gs[1, 1])

        # adatok
        self.centers = centers
        self.lines = lines
        self.clock_dict = clock_dict

        super(MyMplCanvas, self).__init__(fig)
        self.setParent(parent)

        self.plot()

    def plot(self):
        color_list = ['blue', 'green', 'yellow', 'gray', 'gray']

        # subplot 1 es 2: hough vonal csoportok
        for label, line_group in self.lines.items():

            for line in line_group:
                rho, theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = -1*int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = -1*int(y0 - 1000*(a))

                self.axes1.plot([x1, x2], [y1, y2], color=color_list[label], label=f'group_{label}')
                self.axes2.scatter(rho, theta, color=color_list[label], label=f'group_{label}')
            
            rho_c, theta_c = self.centers[label]
            a = np.cos(theta_c)
            b = np.sin(theta_c)
            x0 = a*rho_c
            y0 = b*rho_c
            x1 = int(x0 + 1000*(-b))
            y1 = -1*int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = -1*int(y0 - 1000*(a))

            self.axes1.plot([x1, x2], [y1, y2], color='red')
            self.axes2.scatter(rho_c, theta_c, color='red', s=100, marker='x')
        
        self.axes1.scatter(self.clock_dict['center'][0], -self.clock_dict['center'][1], color='red', s=100, marker='x', linewidths=2)
        for hand in ['hour', 'minute', 'second']:
            if self.clock_dict[hand] == None:
                continue
            self.axes1.scatter(self.clock_dict[hand]['end_point'][0], -self.clock_dict[hand]['end_point'][1], color='red', s=100, marker='x')

        self.axes1.set_xlim([0, 512])
        self.axes1.set_ylim([-512, 0])
        self.axes1.set_aspect('equal', 'box')
        self.axes2.set_xlabel('rho')
        self.axes2.set_ylabel('theta')

        # sublot 3: eredmenyek tablazatosan
        col_labels = ['', 'hour', 'minute', 'second']
        a_row = ['Azimuth (deg)']
        l_rowv = ['Length (px)']
        w_row = ['Width (px)']
        p_row = ['Tip coords']

        for col in ['hour', 'minute', 'second']:
            if self.clock_dict[col] == None:
                continue
            a_row.append(f'{self.clock_dict[col]["azimuth"]:.1f}')
            l_rowv.append(f'{self.clock_dict[col]["length"]:.2f}')
            w_row.append(f'{self.clock_dict[col]["width"]}')
            p_row.append(f'({int(self.clock_dict[col]["end_point"][0])}, {int(self.clock_dict[col]["end_point"][1])})')
        
        cell_text = [a_row, l_rowv, w_row, p_row]
        self.axes3.axis('tight')
        self.axes3.axis('off')
        self.axes3.table(cellText=cell_text, colLabels=col_labels, cellLoc = 'center', loc='center')


# Ablak az eredmenyek megjelenitesere
class ResultDashboardWindow(QWidget):
    def __init__(self, centers, lines, clock_dict):
        super().__init__()
        self.setWindowTitle("Analysing results")
        layout = QVBoxLayout(self)
        self.canvas = MyMplCanvas(centers, lines, clock_dict, self)
        layout.addWidget(self.canvas)

