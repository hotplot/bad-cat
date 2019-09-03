import datetime, sys

import cv2
import numpy as np


def preprocess(frame, roi_coords):
    """Preprocesses the input frame by extracting the ROI, resizing to 224x224,
    converting to grayscale, blurring and equalising the histogram.
    
    Returns both the resized, cropped ROI and the fully-preprocessed ROI as a tuple."""
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed_roi = cv2.GaussianBlur(roi, (13, 13), 0)
    processed_roi = cv2.equalizeHist(processed_roi)
    return roi, processed_roi


def extract_hull(curr_roi, prev_roi):
    """Determines where movement has occurred in `curr_roi` by comparing it to `prev_roi`,
    and forms a complex hull around the region containing movement.

    Returns a `(hull, area)` tuple containing the hull contour and a float representing the
    proportion of the image occupied by the hull."""

    # Find the regions that differ significantly from the previous ROI
    delta = cv2.absdiff(curr_roi, prev_roi)
    _, thresh = cv2.threshold(delta, 64, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda c: cv2.contourArea(c) > 50, contours))

    if len(contours) == 0:
        return None, 0.0

    # Find the convex hull surrounding the movement in the current frame
    points = np.vstack(contours).reshape(-1, 2)
    hull = cv2.convexHull(points)

    # Compute the proportion of the image occupied by the hull
    image_area = curr_roi.shape[0] * curr_roi.shape[1]
    hull_area = cv2.contourArea(hull)
    hull_proportion = hull_area / image_area

    return hull, hull_proportion


def extract_histograms(roi, hull):
    """Computes histograms of the pixel values for the region inside the hull,
    and for the region outside the hull. The histograms are scaled to the range 0..1.
    
    Returns a (foreground hist, background hist) tuple."""

    mask = np.zeros(roi.shape[:2], np.uint8)

    if hull is not None:
        cv2.drawContours(mask, [hull], -1, 255, -1)
    
    inverted_mask = cv2.bitwise_not(mask)

    fg_hist = cv2.calcHist([roi], [0], mask, [32], [0,256]).reshape(-1)
    bg_hist = cv2.calcHist([roi], [0], inverted_mask, [32], [0,255]).reshape(-1)

    if np.max(fg_hist) > 0:
        fg_hist = fg_hist / np.max(fg_hist)
    
    if np.max(bg_hist) > 0:
        bg_hist = bg_hist / np.max(bg_hist)

    return fg_hist, bg_hist


def display_preview(roi, hull, hull_proportion):
    output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    output = cv2.putText(output, f'Hull proportion: {hull_proportion}', (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    if hull is not None:
        output = cv2.drawContours(output.copy(), [hull], -1, (0,0,255))
    cv2.imshow('Output', output)
    if cv2.waitKey(0) == ord('q'):
        exit()


def __log(message, level, stream):
    time_str = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    stream.write(f'[{level}] {time_str}: {message}\n')
    stream.flush()


def log_info(message):
    __log(message, 'INFO', sys.stdout)


def log_error(message):
    __log(message, 'ERROR', sys.stderr)
