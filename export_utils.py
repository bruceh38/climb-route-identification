import os
import cv2
import json
import numpy as np
from typing import List, Dict


def export_transparent_png(rgb: np.ndarray, mask: np.ndarray, out_path: str) -> str:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return out_path


def export_components_csv(components: List[Dict], out_path: str) -> str:
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["area", "centroid_x", "centroid_y", "x", "y", "w", "h", "circularity"])
        for c in components:
            cx, cy = c['centroid']
            x, y, w, h = c['bbox']
            writer.writerow([c['area'], cx, cy, x, y, w, h, c['circularity']])
    return out_path


def write_thresholds_json(lower_hsv: np.ndarray, upper_hsv: np.ndarray, out_path: str, meta: Dict = None) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'lower_hsv': lower_hsv.tolist() if lower_hsv is not None else None,
        'upper_hsv': upper_hsv.tolist() if upper_hsv is not None else None,
    }
    if meta:
        payload.update(meta)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


