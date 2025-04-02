import cv2
import numpy as np
from .utils import load_image

def split_form_with_handwritten_bounding_box(image_path, word_lines):
    """
    Use handwritten word bounding boxes to extract:
    - computer_written: region above the handwritten block
    - hand_written: region from top of handwriting to the bottom of the image
    """
    image = load_image(image_path)
    height, width = image.shape[:2]

    # Step 1: Parse bounding boxes from words.txt
    boxes = []
    for line in word_lines:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) < 9 or parts[1] != "ok":
            continue
        x, y, w, h = map(int, parts[3:7])
        boxes.append((y, y + h))

    if not boxes:
        raise ValueError("No valid handwritten word bounding boxes found.")

    # Step 2: Define split point with top margin
    hand_top_y = max(0, min(y1 for y1, _ in boxes) - 20)

    # Step 3: Crop
    computer_written = image[0:hand_top_y, :]
    hand_written = image[hand_top_y:, :]

    return {
        "computer_written": computer_written,
        "hand_written": hand_written,
        "lines": [hand_top_y]
    }



def detect_horizontal_lines(image, debug=False):
    """
    Try to detect horizontal form lines.
    If detection fails, fall back to estimated line positions.
    """
    img_height, img_width = image.shape[:2]
    debug_image = image.copy()

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological filtering to highlight horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width // 2, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_lines = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0.4 * img_width and h < 10:
            horizontal_lines.append(y)
            if debug:
                cv2.line(debug_image, (0, y), (img_width, y), (0, 0, 255), 2)

    horizontal_lines = sorted(list(set(horizontal_lines)))

    # If failed, fallback to estimated positions
    if len(horizontal_lines) < 2:
        print("[WARNING] Failed to detect horizontal lines. Falling back to estimated positions.")
        est_y1 = int(img_height * 0.08)
        est_y2 = int(img_height * 0.19)
        est_y3 = int(img_height * 0.75)
        horizontal_lines = [est_y1, est_y2, est_y3]
        if debug:
            print(f"[WARNING] Fallback to estimated lines: {horizontal_lines}")
            for y in horizontal_lines:
                cv2.line(debug_image, (0, y), (img_width, y), (0, 255, 255), 2)  # Yellow fallback

    if debug:
        cv2.imwrite("outputs/debug_lines_fallback.png", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
