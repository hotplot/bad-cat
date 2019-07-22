import collections, logging, queue, time

import cv2
import keras
import numpy as np


def preprocess(frame, image_size, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, image_size)
    X = roi.reshape(1, *roi.shape)
    X = keras.applications.mobilenet_v2.preprocess_input(X)
    return X


def drain(frame_queue):
    object = None

    num_discarded = -1
    while True:
        try:
            object = frame_queue.get_nowait()
            num_discarded += 1
        except queue.Empty:
            break

    return object, num_discarded


def classify_frames(model_path, labels, image_size, roi_coords, frame_queue, has_started, should_stop, black_cat_detected):
    # Load the model, then signal that the classifier has started
    model = keras.models.load_model(model_path)
    has_started.set()

    # Set up class state tracking
    current_class = None
    ys = collections.deque(maxlen=2*23)

    # Loop over the supplied frames and process detections
    while not should_stop.is_set():
        # Get the most recent frame, discarding any others that were queued up
        frame, _ = drain(frame_queue)

        if frame is None:
            time.sleep(1.0 / 25.0)
            continue
        
        # Crop and process the frame, run the classifier and add the result to the
        # set of recent predictions
        X = preprocess(frame, image_size, roi_coords)
        y = model.predict(X)[0]
        ys.append(y)

        # Determine the most likely class from the average of the predictions over
        # the last N frames, and signal if it has changed
        predicted_class = np.mean(ys, axis=0).argmax()
        if predicted_class != current_class:
            current_class = predicted_class
            logging.info(f'Detected class changed to {labels[current_class]}')
            if labels[predicted_class] == 'black-cat':
                black_cat_detected.set()
