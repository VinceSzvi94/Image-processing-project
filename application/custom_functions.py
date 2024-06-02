import numpy as np
import cv2

def edgetrim(img: np.ndarray, colorlim=5) -> tuple:
    """
    Egyszinu reszek eltavolitasa szurkearnyalatos reszrol. Nagy colorlim-mel zajos hatter is kezelheto, de az objektum egy resze is elveszhet.
    """
    maxcol = 0
    mincol = 0
    idx = 0

    # felso
    while maxcol - mincol < colorlim:
        toprow = img[idx, :]
        maxcol = np.max(toprow)
        mincol = np.min(toprow)
        top = idx
        idx += 1
    
    maxcol = 0
    mincol = 0
    idx = img.shape[1] - 1

    # jobb
    while maxcol - mincol < colorlim:
        rightrow = img[:, idx]
        maxcol = np.max(rightrow)
        mincol = np.min(rightrow)
        right = idx
        idx -= 1
    
    maxcol = 0
    mincol = 0
    idx = img.shape[0] - 1

    # also
    while maxcol - mincol < colorlim:
        bottomrow = img[idx, :]
        maxcol = np.max(bottomrow)
        mincol = np.min(bottomrow)
        bottom = idx
        idx -= 1
    
    maxcol = 0
    mincol = 0
    idx = 0

    # bal
    while maxcol - mincol < colorlim:
        leftrow = img[:, idx]
        maxcol = np.max(leftrow)
        mincol = np.min(leftrow)
        left = idx
        idx += 1
    
    return (top, right, bottom, left)


def apply_circular_mask(shape: tuple, center: tuple, radius: int) -> np.ndarray:
    """
    generates circular mask
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = np.sqrt((x-center[0])**2 + (y-center[1])**2) <= radius

    return mask