import argparse, collections, datetime, glob, logging, os, pickle, time, traceback
import multiprocessing as mp

from .classify import classify_frames
from .capture import capture_frames


logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.INFO)


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--model', required=True, help='the tflite model to use for classification')
    ap.add_argument('--labels', required=True, help='the pickled labels file')
    ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
    ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')
    ap.add_argument('--video_width', type=int, default=1280, help='the camera horizontal video resolution')
    ap.add_argument('--video_height', type=int, default=720, help='the camera vertical video resolution')
    ap.add_argument('--framerate', type=int, default=5, help='the rate at which to classify frames')
    return vars(ap.parse_args())


def main():
    args = parse_args()

    # Compute the coordinates of the region of interest
    roi = {
        'x': args['roi_x'],
        'y': args['roi_y'],
        'w': args['roi_width'],
        'h': args['roi_height']
    }

    # Determine class labels
    with open(args['labels'], 'rb') as f:
        labels = pickle.load(f)

    # Setup processes

    motion_detected = mp.Event()
    bad_cat_detected = mp.Event()
    classifier_started = mp.Event()
    classification_queue = mp.Queue()
    should_stop = mp.Event()

    capture_proc = mp.Process(
        target=capture_frames, 
        kwargs={
            'roi': roi,
            'capture_size': (args['video_width'], args['video_height']),
            'classification_interval': 1.0 / args['framerate'],
            'classification_queue': classification_queue,
            'should_stop': should_stop,
            'motion_detected': motion_detected
        }
    )

    classify_proc = mp.Process(
        target=classify_frames,
        kwargs={
            'model_path': args['model'],
            'predictions_to_avg': 2 * args['framerate'],
            'labels': labels,
            'classification_queue': classification_queue,
            'has_started': classifier_started,
            'should_stop': should_stop,
            'motion_detected': motion_detected,
            'bad_cat_detected': bad_cat_detected
        }
    )

    # Start the processes.
    # Start with the classifier, and once the model has been loaded start streaming.

    try:
        classify_proc.start()

        while not classifier_started.is_set():
            classifier_started.wait(timeout=0.1)
        
        capture_proc.start()

        logging.info(f'Classifier PID: {classify_proc.pid}')
        logging.info(f'Frame Capture PID: {capture_proc.pid}')

        # Monitor the processes and exit if one of them crashes
        while not should_stop.is_set():
            should_stop.wait(timeout=0.1)
            for proc in [capture_proc, classify_proc]:# , notify_proc]:
                if not proc.is_alive() and not should_stop.is_set():
                    logging.error(f'Process {proc.pid} crashed; exiting')
                    should_stop.set()
    except:
        should_stop.set()
    finally:
        # Wait for the various processes to terminate
        capture_proc.join()
        classify_proc.join()


if __name__ == '__main__':
    main()
