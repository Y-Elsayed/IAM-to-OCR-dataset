import cv2
import numpy as np
from utils import load_image

def detect_horizontal_lines(image, min_line_length=400):
    """Detect the horizontal lines using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min_line_length, maxLineGap=20)

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # horizontal
                horizontal_lines.append((y1 + y2) // 2)

    horizontal_lines = sorted(list(set(horizontal_lines))) # Remove duplicates and sort to have them in order from top to bottom
    return horizontal_lines[:3]

def split_form_into_sections(image_path):
    """Split image into label (computer-written), middle (handwriting), and bottom sections (name)."""
    image = load_image(image_path)
    lines = detect_horizontal_lines(image)

    if len(lines) < 2:  # at least 2 lines needed
        raise ValueError("Could not detect at least 2 horizontal lines.")

    y1, y2 = lines[0], lines[1]
    height = image.shape[0]
    computer_text_crop = image[y1:y2]

    if len(lines) >= 3:
        y3 = lines[2]
        handwritten_text_crop = image[y2:y3]
        bottom_crop = image[y3:height]
        line_coords = [y1, y2, y3]
    else:
        handwritten_text_crop = image[y2:height]
        bottom_crop = None
        line_coords = [y1, y2]

    return {
        "computer_written": computer_text_crop,
        "hand_written": handwritten_text_crop,
        "bottom": bottom_crop,
        "lines": line_coords
    }
