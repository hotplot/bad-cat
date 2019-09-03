import argparse, collections, datetime, glob, os, pickle, time, traceback

import cv2
import keras
import numpy as np

from tensorflow.contrib.lite.python.interpreter import Interpreter

from ..utils import preprocess, extract_hull, extract_histograms


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--video', required=True, help='')
    ap.add_argument('--model', required=True, help='the tflite model to use for classification')
    ap.add_argument('--labels', required=True, help='the pickled labels file')
    ap.add_argument('--nth', type=int, default=4, help='only consider every Nth frame')
    ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
    ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')
    ap.add_argument('--model_width', type=int, default=224, help='the model input width')
    ap.add_argument('--model_height', type=int, default=224, help='the model input height')
    return vars(ap.parse_args())


def main():
    args = parse_args()

    # Compute the coordinates of the region of interest
    roi_coords = (
        args['roi_x'],
        args['roi_y'],
        args['roi_x'] + args['roi_width'],
        args['roi_y'] + args['roi_height']
    )

    # Determine class labels
    with open(args['labels'], 'rb') as f:
        labels = pickle.load(f)

    # Load the model
    interpreter = Interpreter(args['model'])
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Open the video and process each frame
    index = 0
    ys = collections.deque(maxlen=15)

    vc = cv2.VideoCapture(args['video'])
    while vc.isOpened():
        # Read the next frame
        ret, frame = vc.read()
        if ret is False or frame is None:
            break
        
        if index % args['nth'] != 0:
            index += 1
            continue

        # Preprocess the full size frame
        curr_roi, _ = preprocess(frame, roi_coords)

        # Run the classifier and add the result to the set of recent predictions
        X = cv2.cvtColor(curr_roi, cv2.COLOR_GRAY2BGR)
        X = np.reshape(X, (1, *X.shape))
        X = keras.applications.mobilenet_v2.preprocess_input(X)
        
        interpreter.set_tensor(input_tensor_index, X)
        interpreter.invoke()

        y = interpreter.get_tensor(output_tensor_index)[0]
        ys.append(y)
        
        # Determine the instantaneous and average prediction results
        class_index = y.argmax()
        class_name = labels[class_index]

        avg_class_index = np.mean(ys, axis=0).argmax()
        avg_class_name = labels[avg_class_index]

        # Draw the results on the image
        curr_roi = cv2.cvtColor(curr_roi, cv2.COLOR_GRAY2BGR)
        curr_roi = cv2.putText(curr_roi, f'{class_name}: {100*y.max():0.2f}%', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))
        curr_roi = cv2.putText(curr_roi, f'{avg_class_name}: {100*np.mean(ys, axis=0).max():0.2f}%', (25,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))

        cv2.imshow('Prediction', curr_roi)
        if cv2.waitKey(0) == ord('q'):
            break

        # Book keeping for next frame
        index += 1

    vc.release()


if __name__ == '__main__':
    main()
