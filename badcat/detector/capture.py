import datetime, threading, time, os

import cv2
import keras
import numpy as np
import picamera

from ..utils import extract_roi, preprocess_roi, extract_hull, log_info


def __process_frame(frame, crop_coords, classification_queue):
    """Extracts and preprocesses the ROI from the current frame and sends it to the classifier."""

    # Preprocess the current frame
    roi = extract_roi(frame, crop_coords)
    roi, _ = preprocess_roi(roi)

    # Create the input tensor and send it to the classifier
    X = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    X = np.reshape(X, (1, *X.shape))
    X = keras.applications.mobilenet_v2.preprocess_input(X)

    classification_queue.put(X)


def __capture_frames(roi, capture_size, classification_interval, classification_queue, should_stop, predictions, labels, output_dir):
    """Starts streaming from the RPi camera and forwards processed images
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
        next_frame_time = time.time()
        event_storage_timer = None

        output = np.zeros((capture_size[0] * capture_size[1] * 3), dtype=np.uint8)

        for frame in camera.capture_continuous(output, format='bgr', use_video_port=True):
            # Process the current frame
            output = np.reshape(output, (capture_size[1], capture_size[0], 3))
            __process_frame(output, crop_coords, classification_queue)

            # Exit if requested
            if should_stop.is_set():
                break

            # Check whether there has been a motion event worth saving
            class_predictions = zip(predictions, labels)
            best_prediction = sorted(class_predictions)[-1]
            probability, predicted_class = best_prediction

            if output_dir is not None:
                if event_storage_timer is None and predicted_class != 'none' and probability > 0.85:
                    date_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    path = os.path.join(output_dir, f'{date_str}-{predicted_class}.h264')
                    event_storage_timer = threading.Timer(20.0, lambda: buffer.copy_to(path))
                    event_storage_timer.start()
                elif event_storage_timer and not event_storage_timer.is_alive() and predicted_class == 'none':
                    event_storage_timer = None
            
            # Sleep until approximately when the next frame is due to be captured
            curr_time = time.time()
            next_frame_time = max(next_frame_time + classification_interval, curr_time)
            time.sleep(next_frame_time - curr_time)
    
        camera.stop_recording()
        classification_queue.cancel_join_thread()


def start_capturing(**kwargs):
    should_stop = kwargs['should_stop']
    try:
        __capture_frames(**kwargs)
    except KeyboardInterrupt:
        should_stop.set()
