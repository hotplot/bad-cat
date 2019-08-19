import argparse, glob, os.path, pickle

import cv2
import numpy as np

from frameselector import FrameSelector


# Parse arguments
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('-i', '--input', default='train_movies')
ap.add_argument('-o', '--output', default='train_images')
ap.add_argument('-m', '--moving_threshold', type=float, default=0.15)
ap.add_argument('-s', '--still_threshold', type=float, default=0.025)
ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')

args = vars(ap.parse_args())


# Load pickled 'reference frame' info
ref_frame_indexes = {}
try:
    with open('ref_frame_indexes.pickle', 'rb') as f:
        ref_frame_indexes = pickle.load(f)
except:
    pass


def get_reference_frame(path):
    if path in ref_frame_indexes:
        vc = cv2.VideoCapture(path)
        vc.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_indexes[path])
        _, frame = vc.read()
        return frame
    
    selector = FrameSelector(path)
    index, frame = selector.select_frame()

    ref_frame_indexes[path] = index
    with open('ref_frame_indexes.pickle', 'wb') as f:
        pickle.dump(ref_frame_indexes, f)
    
    return frame


def save_image(path, image):
    folder = os.path.split(path)[0]
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, image)


def matches_in(path, glob_str):
    return glob.glob(os.path.join(path, glob_str))


def roi_movement(base_roi, roi):
    # Find the number of pixels that differ significantly from the base image
    delta = cv2.absdiff(base_roi, roi)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum([cv2.contourArea(c) for c in contours])

    # Determine what percentage of the image is moving
    moving_proportion = area / (roi.shape[0] * roi.shape[1])

    # Classify the amount of movement in the ROI
    if moving_proportion > args['moving_threshold']:
        return 'lots'
    if moving_proportion > args['still_threshold']:
        return 'some'
    else:
        return 'none'


def process_video(path):
    # Compute the coordinates of the region of interest
    x1 = args['roi_x']
    x2 = x1 + args['roi_width']
    y1 = args['roi_y']
    y2 = y1 + args['roi_height']

    # Helper function to preprocess images for motion detection
    def preprocess(image):
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.GaussianBlur(processed, (21, 21), 0)
        return processed

    # Prompt for a 'base' frame containing no object
    base_frame = get_reference_frame(path)
    base_roi = preprocess(base_frame[y1:y2, x1:x2])

    # Compute the base output path to use when saving frames
    category = path.split(os.sep)[-2]
    filename = os.path.splitext(os.path.basename(path))[0]
    base_output_path = os.path.join(args['output'], category, filename)

    # Open the video and process each frame
    vc = cv2.VideoCapture(path)

    index = 0

    num_moving_frames = 0
    stationary_frames = []
    
    while vc.isOpened():
        # Read the frame and crop out the ROI
        ret, frame = vc.read()
        if ret is False or frame is None:
            break
        
        roi = frame[y1:y2, x1:x2]
        preprocessed_roi = preprocess(roi)

        # Check how much movement occurs in this frame, and save the ROI
        movement_level = roi_movement(base_roi, preprocessed_roi)
        if movement_level == 'lots':
            output_path = os.path.join(args['output'], category, f'{filename}-{index}.jpg')
            save_image(output_path, roi)
            num_moving_frames += 1
        elif movement_level == 'none':
            stationary_frames.append(roi)
        
        index += 1
    
    # Save the frames where no movement occurred, but only enough to equal the number of 
    # frames where movement *did* occur.
    # Normally there are far more still frames than moving frames, and saving them all 
    # results in significantly unbalanced class sizes.
    num_saved = 0
    while num_saved < num_moving_frames and num_saved < len(stationary_frames):
        output_path = os.path.join(args['output'], 'none', f'{filename}-{index}.jpg')
        save_image(output_path, stationary_frames[num_saved])
        num_saved += 1
        index += 1

    vc.release()


def process_category(path):
    video_paths = matches_in(path, '*.mp4')
    for path in video_paths:
        process_video(path)


def main():
    input_category_paths = matches_in(args['input'], '*')
    for path in input_category_paths:
        process_category(path)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
