import cv2
import numpy as np
from typing import Tuple


def kmeans_hsv(
    hsv_img: np.ndarray,
    k: int = 6,
    max_width: int = 640,
    max_pixels: int = 75000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run OpenCV kmeans on HSV image. Returns (centers_uint8[K,3], labels[h,w] or None, counts[K])."""
    if hsv_img is None:
        return None, None, None
    h, w = hsv_img.shape[:2]
    img_small = hsv_img
    if w > max_width:
        scale = max_width / w
        img_small = cv2.resize(hsv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    data = img_small.reshape(-1, 3).astype(np.float32)
    if data.shape[0] > max_pixels:
        idx = np.random.choice(data.shape[0], max_pixels, replace=False)
        data = data[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compact, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers_u8 = centers.astype(np.uint8)
    # Approx counts from subsample
    uniq, cnts = np.unique(labels, return_counts=True)
    counts = np.zeros(k, dtype=int)
    for u, c in zip(uniq, cnts):
        counts[int(u)] = int(c)
    return centers_u8, None, counts


def bounds_from_center(center_hsv: np.ndarray, margin_h: int = 10, margin_sv: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    c = center_hsv.astype(int)
    lower = np.array([max(0, c[0] - margin_h), max(0, c[1] - margin_sv), max(0, c[2] - margin_sv)], dtype=np.uint8)
    upper = np.array([min(179, c[0] + margin_h), min(255, c[1] + margin_sv), min(255, c[2] + margin_sv)], dtype=np.uint8)
    return lower, upper


