import argparse, collections, datetime, glob, os, pickle, time, traceback

import cv2
import keras
import numpy as np

from tensorflow.contrib.lite.python.interpreter import Interpreter

from .utils import iter_frames


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--video', required=True, help='')
    ap.add_argument('--model', required=True, help='the tflite model to use for classification')
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

    # Load the model
    interpreter = Interpreter(args['model'])
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Open the video and process each frame
    start_time = time.time()
    for index, curr_roi, _ in iter_frames(args['video'], roi_coords):
        X = cv2.cvtColor(curr_roi, cv2.COLOR_GRAY2BGR)
        X = np.reshape(X, (1, *X.shape))
        X = keras.applications.mobilenet_v2.preprocess_input(X)
        
        interpreter.set_tensor(input_tensor_index, X)
        interpreter.invoke()
    
    # Print average inference time
    elapsed_time = time.time() - start_time
    print(f'Processed {index} frames')
    print(f'Time taken: {elapsed_time:0.2f} s')
    print(f'Time per frame: {elapsed_time / index:0.2f} s')
    print(f'Average FPS: {index / elapsed_time:0.1f}')


if __name__ == '__main__':
    main()
