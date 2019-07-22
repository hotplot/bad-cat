import argparse, collections, datetime, glob, logging, os, time, traceback
import multiprocessing as mp

import numpy as np
import keras

from classify import classify_frames
from stream import stream_frames
from notify import notify


logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.INFO)


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--stream', required=True, help='the URL to stream from')
    ap.add_argument('--model', required=True, help='the Keras model to use for classification')
    ap.add_argument('--center_x', type=int, default=775, help='the center X coordinate of the ROI')
    ap.add_argument('--center_y', type=int, default=335, help='the center Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=300, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=300, help='the height of the ROI')
    ap.add_argument('--model_width', type=int, default=224, help='the model input width')
    ap.add_argument('--model_height', type=int, default=224, help='the model input height')
    ap.add_argument('--mqtt_host', required=True, help='the MQTT host to connect to')
    ap.add_argument('--mqtt_port', type=int, default=1883, help='the port on the MQTT host to connect to')
    ap.add_argument('--mqtt_user', default=None, help='the MQTT username to connect with')
    ap.add_argument('--mqtt_pass', default=None, help='the password to connect with')
    ap.add_argument('--mqtt_topic', default=None, help='the MQTT topic to publish detection notifications to')

    return vars(ap.parse_args())


def compute_roi_coords(args):
    x1 = args['center_x'] - args['roi_width']//2
    y1 = args['center_y'] - args['roi_height']//2
    x2 = x1 + args['roi_width']
    y2 = y1 + args['roi_height']
    return (x1, y1, x2, y2)


def main():
    args = parse_args()

    # Compute the coordinates of the region of interest
    roi_coords = compute_roi_coords(args)

    # Determine class labels
    class_names = sorted(glob.glob('train_images/*'))
    class_names = [ os.path.split(n)[1] for n in class_names ]
    labels = dict(zip(range(len(class_names)), class_names))

    # Setup processes

    black_cat_detected = mp.Event()
    classifier_started = mp.Event()
    should_stop = mp.Event()
    frame_queue = mp.Queue()

    stream_proc = mp.Process(
        target=stream_frames, 
        kwargs={
            'url': args['stream'],
            'roi_coords': roi_coords,
            'frame_queue': frame_queue,
            'should_stop': should_stop
        }
    )

    classify_proc = mp.Process(
        target=classify_frames,
        kwargs={
            'model_path': args['model'],
            'labels': labels,
            'image_size': (args['model_width'], args['model_height']),
            'roi_coords': roi_coords,
            'frame_queue': frame_queue,
            'has_started': classifier_started,
            'should_stop': should_stop,
            'black_cat_detected': black_cat_detected
        }
    )

    notify_proc = mp.Process(
        target=notify,
        kwargs={
            'mqtt_host': args['mqtt_host'],
            'mqtt_port': args['mqtt_port'],
            'mqtt_user': args['mqtt_user'],
            'mqtt_pass': args['mqtt_pass'],
            'mqtt_topic': args['mqtt_topic'],
            'should_publish': black_cat_detected,
            'should_stop': should_stop,
        }
    )

    # Start processes.
    # Start with the classifier, and once the model has been loaded start streaming.

    classify_proc.start()
    notify_proc.start()

    while not classifier_started.is_set():
        classifier_started.wait(timeout=0.1)
    
    stream_proc.start()

    logging.info(f'Classifier PID: {classify_proc.pid}')
    logging.info(f'Streamer PID: {stream_proc.pid}')
    logging.info(f'Notifier PID: {notify_proc.pid}')

    # Monitor the processes and exit if one of them crashes
    while not should_stop.is_set():
        should_stop.wait(timeout=0.1)
        for proc in [stream_proc, classify_proc, notify_proc]:
            if not proc.is_alive() and not should_stop.is_set():
                logging.error(f'Process {proc.pid} crashed; exiting')
                should_stop.set()

    # Wait for the various processes to terminate
    stream_proc.join()
    classify_proc.join()
    notify_proc.join()


if __name__ == '__main__':
    main()
