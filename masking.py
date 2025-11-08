import cv2
import numpy as np
from typing import Tuple


def build_mask_hsv(hsv_img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if hsv_img is None or lower is None or upper is None:
        return None
    return cv2.inRange(hsv_img, lower, upper)


def morphology_cleanup(
    mask: np.ndarray,
    median_blur_size: int = 0,
    open_size: int = 3,
    close_size: int = 7,
    do_open: bool = True,
    do_close: bool = True,
) -> np.ndarray:
    if mask is None:
        return None
    out = mask.copy()
    if median_blur_size and median_blur_size % 2 == 1:
        out = cv2.medianBlur(out, median_blur_size)
    if do_open:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k1, iterations=1)
    if do_close:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k2, iterations=1)
    return out


def remove_small_components(mask: np.ndarray, area_min: int = 150) -> np.ndarray:
    if mask is None:
        return None
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out


