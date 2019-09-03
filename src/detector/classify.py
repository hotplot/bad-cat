import collections, datetime, queue, time

import numpy as np

from tensorflow.lite.python.interpreter import Interpreter

from ..utils import log_info


def __drain(classification_queue):
    """Returns the last object in `classification_queue` and discards the rest"""

    object = None
    while True:
        try:
            object = classification_queue.get_nowait()
        except queue.Empty:
            break
    return object


def __classify_frames(model_path, predictions_to_avg, classification_queue, has_started, should_stop, predictions, labels):
    # Load the model, then signal that the classifier has started
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    has_started.set()

    # Set up class state tracking
    ys = collections.deque(maxlen=predictions_to_avg)

    # Loop over the supplied frames and process detections
    while not should_stop.is_set():
        # Get the next input and prepare it for classification
        X = __drain(classification_queue)
        if X is None:
            time.sleep(1.0 / 25.0)
            continue

        # Run the classifier and add the result to the set of recent predictions
        interpreter.set_tensor(input_tensor_index, X)
        interpreter.invoke()

        y = interpreter.get_tensor(output_tensor_index)[0]
        ys.append(y)

        # Update the shared predictions using the average of recent predictions
        if len(ys) == predictions_to_avg:
            predictions[:] = np.mean(ys, axis=0)


def start_classifying(**kwargs):
    should_stop = kwargs['should_stop']
    try:
        __classify_frames(**kwargs)
    except KeyboardInterrupt:
        should_stop.set()
