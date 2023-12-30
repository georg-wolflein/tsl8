import numpy as np
import cv2


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb, np.array([0.299, 0.587, 0.114])).astype(
        np.uint8
    )  # see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html


@np.vectorize(signature="(n,m,c)->()")
def is_background(patch: np.ndarray) -> bool:
    patch = rgb2gray(patch)

    # hardcoded thresholds
    edge = cv2.Canny(patch, 40, 100)

    # avoid dividing by zero
    edge = (edge / np.max(edge)) if np.max(edge) != 0 else 0
    num_pixels = np.prod(patch.shape)
    edge = ((np.sum(np.sum(edge)) / num_pixels) * 100) if num_pixels != 0 else 0

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    return edge < 2.0
