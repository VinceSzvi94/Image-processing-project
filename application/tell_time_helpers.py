import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
# from kneed import KneeLocator
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import cv2

from application.custom_functions import apply_circular_mask

# adatszerk az eredmenyek tarolasara
CLOCK_DICT = {
    "center": None,
    "hour": None,
    "minute": None,
    "second": None
}
HAND_DICT = {
    "azimuth": None,
    "length": None,
    "end_point": None,
    "line": None,
    "width": None # csak a percmutatonal szamit
}


# vonal theta szogvaltas
def line2rad(line):
    rho, theta = line
    return rho, math.radians(theta)

def line2deg(line):
    rho, theta = line
    return rho, math.degrees(theta)


# vonal metszespont
def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    x0, y0 = np.linalg.solve(a, b)
    return x0, y0


# tavolsag ponttol
def dist_to_point(x, y, p):
    return math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)

def dist_to_point_np(arr: np.ndarray, p):
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    return np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)


# tavolsag vonaltol
def dist_to_line(x, y, rho, theta):
    return abs(y * math.cos(theta) + x * math.sin(theta) - rho)

def dist_to_line_np(arr: np.ndarray, rho, theta):
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    return np.abs(x * np.cos(theta) + y * np.sin(theta) - rho)


# vonalhoz megadott tartomanyban van-e (es logikai maszk)
def in_range_of_line(x, y, rho, theta, dist=2):
    return dist_to_line(x, y, rho, theta) < dist

def in_range_of_line_np(arr: np.ndarray, rho, theta, dist=2):
    return dist_to_line_np(arr, rho, theta) < dist


# azimuth irany ponthoz kepest (ora kozeppont, nem feltetlenul kep kozeppont)
def azimuth_to_center(x, y, center):
    azimuth = math.degrees(math.atan2(y - center[0], x - center[1]))
    if azimuth < 0:
        azimuth = azimuth + 360
    return azimuth

def azimuth_to_center_np(arr: np.ndarray, center):
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    azimuth = np.degrees(np.arctan2(x - center[0], y - center[1]))
    azimuth -= 180 # mivel a kep origoja a bal felso sarokban van
    return -azimuth % 360


# azimuth irany szektor tartomanyban van-e (es logikai maszk)
def in_sector(x, y, center, sector):
    """
    sector: (start, end) in DEGREES!!!
    (355, 5) -> (355, 360) + (0, 5)
    """
    azimuth = azimuth_to_center(x, y, center)
    if sector[0] < sector[1]:
        return sector[0] <= azimuth <= sector[1]
    else:
        return azimuth >= sector[0] or azimuth <= sector[1]

def in_sector_np(arr: np.ndarray, center, sector):
    """
    sector: (start, end) in DEGREES!!!
    (355, 5) -> (355, 360) + (0, 5)
    """
    azimuth = azimuth_to_center_np(arr, center)
    if sector[0] < sector[1]:
        return (sector[0] <= azimuth) & (azimuth <= sector[1])
    else:
        return (azimuth >= sector[0]) | (azimuth <= sector[1])


# melyik vonalhoz van a legkozelebb mask
def closest_line_mask(img, target_line, lines):
    dist_to_target = dist_to_line_np(img, target_line[0], target_line[1])
    if len(lines) == 0:
        return np.full(img.shape, True)
    elif len(lines) == 1:
        return dist_to_target <= dist_to_line_np(img, lines[0][0], lines[0][1])
    else: # ha tobb mint 3 vonal van nem mukodik rendesen, elvileg viszont ilyen nem lehet
        dist1 = dist_to_line_np(img, lines[0][0], lines[0][1])
        dist2 = dist_to_line_np(img, lines[1][0], lines[1][1])
        return (dist_to_target <= dist1) & (dist_to_target <= dist2)


# ora kp meghatarozasa
def find_center(img, lines, tol=45):
    img_center = img.shape[0] // 2, img.shape[1] // 2

    if len(lines) == 1:
        return img_center
    
    elif len(lines) == 2:
        intersection = line_intersection(lines[0], lines[1])
        int_x, int_y = intersection
        if dist_to_point(int_x, int_y, img_center) > tol:
            print(f"WARNING: line intersection is outside of center tolerance (tol={tol}px), using image center instead")
            return img_center
        return int(int_x), int(int_y)
    
    elif len(lines) == 3:
        intersection1 = line_intersection(lines[0], lines[1])
        intersection2 = line_intersection(lines[1], lines[2])
        intersection3 = line_intersection(lines[2], lines[0])
        int_x, int_y = int((intersection1[0] + intersection2[0] + intersection3[0]) // 3), int((intersection1[1] + intersection2[1] + intersection3[1]) // 3)
        if dist_to_point(int_x, int_y, img_center) > tol:
            print(f"WARNING: line intersection is outside of center tolerance (tol={tol}px), using image center instead")
            return img_center
        return int(int_x), int(int_y)
    
    else:
        raise ValueError("Number of lines must be 1, 2 or 3")


# mutato vegek meghatarozasa a hozzatartozo meretekkel
def find_hand_ends(img, center, line, dist_tol=5):
    """returns (azimuth, distance, point) for both hand ends"""
    dist_arr = dist_to_point_np(img, center)

    # 1 csak a vonal kornyezete + a kozeppont es vonal kozotti kor latszodjon
    mask1 = in_range_of_line_np(img, line[0], line[1], dist=dist_tol)
    dist1 = dist_to_line(center[0], center[1], line[0], line[1])
    circ_mask = apply_circular_mask(img.shape, center, dist1)
    mask1 = np.where(mask1, 255, 0).astype('uint8')
    mask1 = cv2.bitwise_and(img, mask1)
    mask1[circ_mask] = 255

    # 2 floodfill esetlegesen bentmaradt szamok levagasara
    h, w = img.shape[:2]
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask1, ff_mask, center, 255, 250, 255, cv2.FLOODFILL_FIXED_RANGE)
    mask12 = np.where(ff_mask[1:-1, 1:-1] != 0, True, False)

    # 3 maskok a ket iranyra
    azimuth2 = math.degrees(line[1])
    mask2 = in_sector_np(img, center, ((azimuth2 - 45) % 360, (azimuth2 + 45) % 360))
    mask2 &= mask12
    azimuth3 = (azimuth2 + 180) % 360
    mask3 = in_sector_np(img, center, ((azimuth3 - 45) % 360, (azimuth3 + 45) % 360))
    mask3 &= mask12

    # elso iranyba
    dist_arr2 = dist_arr.copy()
    dist_arr2[~mask2] = 0
    dist2 = np.max(dist_arr2)
    p2 = np.unravel_index(np.argmax(dist_arr2), dist_arr2.shape)
    p2 = p2[::-1] # sor: y oszlop: x
    if dist2 == 0:
        p2 = center
    
    # masodik iranyba
    dist_arr3 = dist_arr.copy()
    dist_arr3[~mask3] = 0
    dist3 = np.max(dist_arr3)
    p3 = np.unravel_index(np.argmax(dist_arr3), dist_arr3.shape)
    p3 = p3[::-1] # sor: y oszlop: x
    if dist3 == 0:
        p3 = center
    
    return (azimuth2, dist2, p2), (azimuth3, dist3, p3)


# mutato vastagsagok kozbenso reszen
def find_hand_widths(img, center, line, other_lines, end_dist, azi, azi_tol=20, r1=60, r2=110):
    if end_dist < r2:
        r2 = end_dist
    
    dist_arr = dist_to_line_np(img, line[0], line[1])

    mask1 = apply_circular_mask(img.shape, center, r1)
    mask2 = apply_circular_mask(img.shape, center, r2)
    mask3 = closest_line_mask(img, line, other_lines) # a szegmentalt kep vonalhoz legkozelebbi resze
    bool_img = img == 255
    mask3 &= bool_img
    mask4 = in_sector_np(img, center, ((azi - azi_tol) % 360, (azi + azi_tol) % 360))

    dist_arr[mask1] = 0
    dist_arr[~mask2] = 0
    dist_arr[~mask3] = 0
    dist_arr[~mask4] = 0

    return np.max(dist_arr)


# mutato tipusok elkulonitese a clustering eredmenyek alapjan
def no_sh_2c(img, labels_used, centers):
    clock_dict = CLOCK_DICT.copy()
    clock_dict["hour"] = HAND_DICT.copy()
    clock_dict["minute"] = HAND_DICT.copy()

    label1, label2 = labels_used
    line1, line2 = centers[label1], centers[label2] # cluster kozeppontok, ezek a mutato iranyokba levo vonalaknak felelnek meg !!

    # ora kozep pontja
    clock_dict["center"] = find_center(img, [line1, line2])
    
    # mutatok, felteve hogy a mutato a mutatas iranyaba mindig hosszabb
    end11, end12 = find_hand_ends(img, clock_dict["center"], line1)
    if end11[1] > end12[1]:
        end1 = end11
    else:
        end1 = end12
    end21, end22 = find_hand_ends(img, clock_dict["center"], line2)
    if end21[1] > end22[1]:
        end2 = end21
    else:
        end2 = end22
    
    # feltetelezve hogy az ora mutato a rovidebb
    if end1[1] < end2[1]:
        clock_dict["hour"]["azimuth"] = end1[0]
        clock_dict["hour"]["length"] = end1[1]
        clock_dict["hour"]["end_point"] = end1[2]
        clock_dict["hour"]["line"] = line1
        clock_dict["minute"]["azimuth"] = end2[0]
        clock_dict["minute"]["length"] = end2[1]
        clock_dict["minute"]["end_point"] = end2[2]
        clock_dict["minute"]["line"] = line2
    else:
        clock_dict["hour"]["azimuth"] = end2[0]
        clock_dict["hour"]["length"] = end2[1]
        clock_dict["hour"]["end_point"] = end2[2]
        clock_dict["hour"]["line"] = line2
        clock_dict["minute"]["azimuth"] = end1[0]
        clock_dict["minute"]["length"] = end1[1]
        clock_dict["minute"]["end_point"] = end1[2]
        clock_dict["minute"]["line"] = line1

    return clock_dict

def no_sh_1c(img, labels_used, centers, hand_ratio_lim=0.5):
    clock_dict = CLOCK_DICT.copy()
    clock_dict["hour"] = HAND_DICT.copy()
    clock_dict["minute"] = HAND_DICT.copy()

    label = labels_used[0]
    line = centers[label] # cluster kozeppont, a mutatok iranyanak felel meg !!

    # ora kozep pontja
    clock_dict["center"] = find_center(img, [line])
    
    # mutatok, end1 a hosszabb iranyba
    end1, end2 = find_hand_ends(img, clock_dict["center"], line)
    if end1[1] < end2[1]:
        end1, end2 = end2, end1
    
    # feltetelezve hogy az ora mutato hosszabb mint a perc mutato fele (vagy a megadott ertek), a mutatok vegi tulnyulas viszont rovidebb
    if end2[1] > hand_ratio_lim*end1[1]:
        clock_dict["hour"]["azimuth"] = end2[0]
        clock_dict["hour"]["length"] = end2[1]
        clock_dict["hour"]["end_point"] = end2[2]
        clock_dict["hour"]["line"] = line
        clock_dict["minute"]["azimuth"] = end1[0]
        clock_dict["minute"]["length"] = end1[1]
        clock_dict["minute"]["end_point"] = end1[2]
        clock_dict["minute"]["line"] = line
    else: # az oramutato is a percig tart igy, abrazolasnal igy vegig egyben lesznek
        clock_dict["hour"]["azimuth"] = end1[0]
        clock_dict["hour"]["length"] = end1[1]
        clock_dict["hour"]["end_point"] = end1[2]
        clock_dict["hour"]["line"] = line
        clock_dict["minute"]["azimuth"] = end1[0]
        clock_dict["minute"]["length"] = end1[1]
        clock_dict["minute"]["end_point"] = end1[2]
        clock_dict["minute"]["line"] = line

    return clock_dict

def sh_3c(img, labels_used, centers):
    clock_dict = CLOCK_DICT.copy()
    clock_dict["hour"] = HAND_DICT.copy()
    clock_dict["minute"] = HAND_DICT.copy()
    clock_dict["second"] = HAND_DICT.copy()

    label1, label2, label3 = labels_used
    line1, line2, line3 = centers[label1], centers[label2], centers[label3] # cluster kozeppontok, ezek a mutato iranyokba levo vonalaknak felelnek meg !!

    # ora kozep pontja
    clock_dict["center"] = find_center(img, [line1, line2, line3])

    # mutatok, felteve hogy a mutato a mutatas iranyaba mindig hosszabb
    end11, end12 = find_hand_ends(img, clock_dict["center"], line1)
    if end11[1] > end12[1]:
        end1 = end11
    else:
        end1 = end12
    end21, end22 = find_hand_ends(img, clock_dict["center"], line2)
    if end21[1] > end22[1]:
        end2 = end21
    else:
        end2 = end22
    end31, end32 = find_hand_ends(img, clock_dict["center"], line3)
    if end31[1] > end32[1]:
        end3 = end31
    else:
        end3 = end32
    
    # masodpercmutato megkeresese, feltetelezve hogy az a legvekonyabb
    # find_hand_widths(img, center, line, other_lines, end_dist, azi, azi_tol=20, r1=60, r2=150)
    w1 = find_hand_widths(img, clock_dict["center"], line1, [line2, line3], end1[1], end1[0])
    w2 = find_hand_widths(img, clock_dict["center"], line2, [line1, line3], end2[1], end2[0])
    w3 = find_hand_widths(img, clock_dict["center"], line3, [line1, line2], end3[1], end3[0])

    if w1 < w2 and w1 < w3:
        clock_dict["second"]["azimuth"] = end1[0]
        clock_dict["second"]["length"] = end1[1]
        clock_dict["second"]["end_point"] = end1[2]
        clock_dict["second"]["line"] = line1
        clock_dict["second"]["width"] = round(2*w1, 2)

        # feltetelezve hogy az ora mutato a rovidebb
        if end3[1] < end2[1]:
            clock_dict["hour"]["azimuth"] = end3[0]
            clock_dict["hour"]["length"] = end3[1]
            clock_dict["hour"]["end_point"] = end3[2]
            clock_dict["hour"]["line"] = line3
            clock_dict["hour"]["width"] = round(2*w3, 2)
            clock_dict["minute"]["azimuth"] = end2[0]
            clock_dict["minute"]["length"] = end2[1]
            clock_dict["minute"]["end_point"] = end2[2]
            clock_dict["minute"]["line"] = line2
            clock_dict["minute"]["width"] = round(2*w2, 2)
        else:
            clock_dict["hour"]["azimuth"] = end2[0]
            clock_dict["hour"]["length"] = end2[1]
            clock_dict["hour"]["end_point"] = end2[2]
            clock_dict["hour"]["line"] = line2
            clock_dict["hour"]["width"] = round(2*w2, 2)
            clock_dict["minute"]["azimuth"] = end3[0]
            clock_dict["minute"]["length"] = end3[1]
            clock_dict["minute"]["end_point"] = end3[2]
            clock_dict["minute"]["line"] = line3
            clock_dict["minute"]["width"] = round(2*w3, 2)

    elif w2 < w1 and w2 < w3:
        clock_dict["second"]["azimuth"] = end2[0]
        clock_dict["second"]["length"] = end2[1]
        clock_dict["second"]["end_point"] = end2[2]
        clock_dict["second"]["line"] = line2
        clock_dict["second"]["width"] = round(2*w2, 2)

        # feltetelezve hogy az ora mutato a rovidebb
        if end3[1] < end1[1]:
            clock_dict["hour"]["azimuth"] = end3[0]
            clock_dict["hour"]["length"] = end3[1]
            clock_dict["hour"]["end_point"] = end3[2]
            clock_dict["hour"]["line"] = line3
            clock_dict["hour"]["width"] = round(2*w3, 2)
            clock_dict["minute"]["azimuth"] = end1[0]
            clock_dict["minute"]["length"] = end1[1]
            clock_dict["minute"]["end_point"] = end1[2]
            clock_dict["minute"]["line"] = line1
            clock_dict["minute"]["width"] = round(2*w1, 2)
        else:
            clock_dict["hour"]["azimuth"] = end1[0]
            clock_dict["hour"]["length"] = end1[1]
            clock_dict["hour"]["end_point"] = end1[2]
            clock_dict["hour"]["line"] = line1
            clock_dict["hour"]["width"] = round(2*w1, 2)
            clock_dict["minute"]["azimuth"] = end3[0]
            clock_dict["minute"]["length"] = end3[1]
            clock_dict["minute"]["end_point"] = end3[2]
            clock_dict["minute"]["line"] = line3
            clock_dict["minute"]["width"] = round(2*w3, 2)

    else:
        clock_dict["second"]["azimuth"] = end3[0]
        clock_dict["second"]["length"] = end3[1]
        clock_dict["second"]["end_point"] = end3[2]
        clock_dict["second"]["line"] = line3
        clock_dict["second"]["width"] = round(2*w3, 2)

        # feltetelezve hogy az ora mutato a rovidebb
        if end2[1] < end1[1]:
            clock_dict["hour"]["azimuth"] = end2[0]
            clock_dict["hour"]["length"] = end2[1]
            clock_dict["hour"]["end_point"] = end2[2]
            clock_dict["hour"]["line"] = line2
            clock_dict["hour"]["width"] = round(2*w2, 2)
            clock_dict["minute"]["azimuth"] = end1[0]
            clock_dict["minute"]["length"] = end1[1]
            clock_dict["minute"]["end_point"] = end1[2]
            clock_dict["minute"]["line"] = line1
            clock_dict["minute"]["width"] = round(2*w1, 2)
        else:
            clock_dict["hour"]["azimuth"] = end1[0]
            clock_dict["hour"]["length"] = end1[1]
            clock_dict["hour"]["end_point"] = end1[2]
            clock_dict["hour"]["line"] = line1
            clock_dict["hour"]["width"] = round(2*w1, 2)
            clock_dict["minute"]["azimuth"] = end2[0]
            clock_dict["minute"]["length"] = end2[1]
            clock_dict["minute"]["end_point"] = end2[2]
            clock_dict["minute"]["line"] = line2
            clock_dict["minute"]["width"] = round(2*w2, 2)
    
    return clock_dict

def sh_2c(img, labels_used, centers):
    # 6 kulon eset: 3 mutato lehet egyedul es minden esetben a maradek 2 nezhet egy iranyba vagy forditva
    # nincs ilyen tesztkep, NINCS IMPLEMENTALVA
    raise TypeError("No implementation for test pic")

def sh_1c(img, labels_used, centers):
    # 4 kulon eset: vagy mindharom egy iranyba nez, vagy 2 nez egy iranyba es a harmadik forditva, osszesen 1+3=4 eset
    # nincs ilyen tesztkep, NINCS IMPLEMENTALVA
    raise TypeError("No implementation for test pic")


# DBSCAN clustering a hough vonalakon
def cluster_lines(lines, eps=0.15):
    data = np.array([[line[0][0], line[0][1]] for line in lines])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    dbscan_i = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan_i.fit_predict(scaled_data)
    n_clusters = len(set(labels))

    centers = [ np.mean(data[labels == i], axis=0) for i in range(n_clusters) ]
    inertia = sum([np.linalg.norm(data[labels == i] - centers[i]) for i in range(n_clusters)])

    labelled_dict = {}
    for label, line in zip(labels, lines):
        if label not in labelled_dict:
            labelled_dict[label] = [line[0]]
        else:
            labelled_dict[label].append(line[0])

    # labelled_dict key = centers index = cluster label
    return labelled_dict, centers, n_clusters, inertia


def process_data(img, lines, second_hand: bool):
    n_hands = 3 if second_hand else 2
    labelled_dict, centers, n_clusters, inertia = cluster_lines(lines)
    
    if n_clusters > n_hands:
        print(f"Too many clusters ({n_clusters}), ones with fewest lines will be removed")
        line_no_dict = {label: len(lines) for label, lines in labelled_dict.items()}
        line_no_dict_sorted = sorted(line_no_dict.items(), key=lambda x: x[1], reverse=True)
        for label, _ in line_no_dict_sorted[n_hands:]:
            del labelled_dict[label]
            print(f"Cluster {label} removed")
    
    labels_used = list(labelled_dict.keys())
    n_clusters = len(labels_used)
    clock_dict = {}

    # mutatok meghatarozasa, sok, osszesen 14 eset!!!
    # nincs tesztkep mindegyikhez !!!
    # az eseteket reszben kulon fuggvenyek kezelik
    if not second_hand and (n_clusters == 2):
        clock_dict = no_sh_2c(img, labels_used, centers)
    elif not second_hand and (n_clusters == 1):
        clock_dict = no_sh_1c(img, labels_used, centers) # 2 kulon eset
    elif second_hand and (n_clusters == 3):
        clock_dict = sh_3c(img, labels_used, centers)
    elif second_hand and (n_clusters == 2):
        clock_dict = sh_2c(img, labels_used, centers) # 6 kulon eset
    elif second_hand and (n_clusters == 1):
        clock_dict = sh_1c(img, labels_used, centers) # 4 kulon eset
    else:
        raise ValueError(f"Unknown case, second_hand={second_hand}, n_clusters={n_clusters}")

    return clock_dict, labelled_dict, centers


# ido megmondasa
def tell_seconds(hand_dict):
    if not hand_dict is None:
        return int(hand_dict["azimuth"] // 6)
    return 0

def tell_minutes(hand_dict):
    return int(hand_dict["azimuth"] // 6)

def tell_hours(hand_dict, minutes: int):
    fractional, integer = math.modf(hand_dict["azimuth"] / 30)
    fractional *= 60

    if abs(fractional - minutes) > 35: # lehetseges pontatlansag egesz ora korul
        if fractional > minutes: # pl: ora alapjan 1.58 de valosagban 2 ora 2 perc
            integer = (integer + 1) % 12
        else: # pl: ora alapjan 1.02 de valosagban 0 ora 58 perc
            integer = (integer - 1) % 12
        
    return int(integer)

def tell_time(clock_dict, second_hand: bool):
    minutes = tell_minutes(clock_dict["minute"])
    minutes_str = str(minutes)
    if len(minutes_str) == 1:
        minutes_str = "0" + minutes_str
    
    hours = str(tell_hours(clock_dict["hour"], minutes))
    if len(hours) == 1:
        hours = "0" + hours

    time_str = hours + ":" + minutes_str

    if second_hand:
        seconds = str(tell_seconds(clock_dict["second"]))
        if len(seconds) == 1:
            seconds = "0" + seconds
        time_str += ":" + seconds
    
    return time_str


