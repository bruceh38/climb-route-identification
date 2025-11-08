import cv2
import numpy as np
from typing import List, Dict


def find_components(mask: np.ndarray, area_min: int = 150) -> List[Dict]:
    if mask is None:
        return []
    components = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < area_min:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        peri = cv2.arcLength(cnt, True)
        circularity = float(4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        components.append({
            'area': area,
            'centroid': (int(cx), int(cy)),
            'bbox': (int(x), int(y), int(w), int(h)),
            'circularity': circularity,
        })
    return components


