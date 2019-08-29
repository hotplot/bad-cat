import collections, logging, queue, time
import datetime

import numpy as np

from tensorflow.lite.python.interpreter import Interpreter


def classify_frames(model_path, predictions_to_avg, labels, classification_queue, has_started, should_stop, motion_detected, bad_cat_detected):
    # Load the model, then signal that the classifier has started
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    has_started.set()

    # Set up class state tracking
    current_class = None
    ys = collections.deque(maxlen=predictions_to_avg)

    # Loop over the supplied frames and process detections
    while not should_stop.is_set():
        # Get the next input and prepare it for classification
        X = classification_queue.get()
        X = np.reshape(X, (1, *X.shape))
        X = np.array(X, dtype=np.float32)

        # Run the classifier and add the result to the set of recent predictions
        interpreter.set_tensor(input_tensor_index, X)
        interpreter.invoke()

        y = interpreter.get_tensor(output_tensor_index)[0]
        ys.append(y)

        # Determine the most likely class from the average of the predictions over
        # the last N frames, and signal if it has changed
        predicted_class = np.mean(ys, axis=0).argmax()
        if predicted_class != current_class:
            current_class = predicted_class
            
            print(f'{datetime.datetime.now().strftime("%X")}: Detected class changed to {labels[current_class]}')
            logging.info(f'Detected class changed to {labels[current_class]}')
            
            # Notify if any motion has been detected
            if labels[predicted_class] != 'none':
                motion_detected.set()
            else:
                motion_detected.clear()

            # Notify if a bad cat has been detected
            if labels[predicted_class] == 'black-cat':
                bad_cat_detected.set()
            else:
                bad_cat_detected.clear()
