import datetime, logging, time, traceback

import cv2
import numpy as np
import picamera

from ..utils import preprocess, extract_hull, extract_histograms


def __process_frame(frame, prev_roi, crop_coords, classification_queue):
    """Extracts the fore and background histograms from the current frame and sends
    them to the classifier"""

    # Preprocess the current frame
    _, curr_roi = preprocess(frame, crop_coords)

    # If we don't have a previous ROI, there is nothing more to do
    if prev_roi is None:
        return curr_roi

    # Extract the hull surrounding the movement in the ROI, if there is any,
    # and compute foreground/background histograms
    hull, hull_proportion = extract_hull(curr_roi, prev_roi)
    fg_hist, bg_hist = extract_histograms(curr_roi, hull)

    # Create the input tensor and send it to the classifier
    X = np.hstack((hull_proportion, fg_hist, bg_hist))
    classification_queue.put(X)

    return curr_roi


def capture_frames(roi, capture_size, classification_interval, classification_queue, should_stop, motion_detected):
    """Starts streaming from the RPi camera and forwards processed histograms
    to the classifier via `classification_queue` until `should_stop` is set."""

    crop_coords = (
        roi['x'],
        roi['y'],
        roi['x'] + roi['w'],
        roi['y'] + roi['h']
    )
    
    with picamera.PiCamera() as camera:
        camera.resolution = capture_size
        camera.framerate = 20

        # Begin recording to the circular buffer (used for saving footage of events)
        buffer = picamera.PiCameraCircularIO(camera, seconds=20)
        camera.start_recording(buffer, format='h264', intra_period=19, sps_timing=True)

        # Begin continuously capturing and sending frames to the classifier
        next_capture_time = time.time()
        next_store_time = None

        prev_roi = None
        output = np.zeros((*capture_size, 3), dtype=np.uint8)

        for frame in camera.capture_continuous(output, format='bgr', use_video_port=True):
            # Process the current frame
            prev_roi = __process_frame(frame, prev_roi, crop_coords, classification_queue)

            # Check whether we should exit, or need to save event footage
            if should_stop.is_set():
                break

            if next_store_time is None and motion_detected.is_set():
                next_store_time = time.time() + 10.0
            elif next_store_time is not None and time.time() >= next_store_time:
                filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.h264')
                buffer.copy_to(filename)
                next_store_time = None
            
            # Sleep until approximately when the next frame is due to be captured
            next_capture_time += classification_interval
            sleep_duration = max(0, next_capture_time - time.time())
            time.sleep(sleep_duration)
    
        camera.stop_recording()                          
