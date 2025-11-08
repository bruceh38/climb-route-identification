import cv2
import numpy as np
from typing import Tuple


def grayworld_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    mean = img.mean(axis=(0, 1))
    gray = float(mean.mean())
    scale = gray / (mean + 1e-6)
    balanced = img * scale[None, None, :]
    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_bilateral(img_rgb: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)


def apply_clahe_v(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)


def preprocess_pipeline(
    img_rgb: np.ndarray,
    use_grayworld: bool = False,
    use_bilateral_filter: bool = False,
    use_clahe_v_enhancement: bool = True,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if img_rgb is None:
        return None, None
    result = img_rgb.copy()
    if scale != 1.0:
        h, w = result.shape[:2]
        result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if use_grayworld:
        result = grayworld_white_balance(result)
    if use_bilateral_filter:
        result = apply_bilateral(result, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    if use_clahe_v_enhancement:
        result = apply_clahe_v(result, clahe_clip_limit, clahe_tile_grid)
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    return result, hsv


