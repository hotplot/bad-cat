import argparse, glob, os.path, pickle

import cv2
import numpy as np

from frameselector import FrameSelector


# Parse arguments
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('-i', '--input', default='train_movies')
ap.add_argument('-o', '--output', default='train_images')
ap.add_argument('-w', '--width', type=int, default=300)
ap.add_argument('-h', '--height', type=int, default=300)
ap.add_argument('-x', '--center_x', type=int, default=775)
ap.add_argument('-y', '--center_y', type=int, default=335)
ap.add_argument('-m', '--moving_threshold', type=int, default=20000)
ap.add_argument('-s', '--still_threshold', type=int, default=5000)

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
    delta = cv2.absdiff(base_roi, roi)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum([cv2.contourArea(c) for c in contours])

    if area > args['moving_threshold']:
        return 'lots'
    if area > args['still_threshold']:
        return 'some'
    else:
        return 'none'


def process_video(path):
    # Compute the coordinates of the region of interest
    startX = args['center_x'] - args['width']//2
    finishX = startX + args['width']
    startY = args['center_y'] - args['height']//2
    finishY = startY + args['height']

    # Helper function to preprocess images for motion detection
    def preprocess(image):
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.GaussianBlur(processed, (21, 21), 0)
        return processed

    # Prompt for a 'base' frame containing no object
    base_frame = get_reference_frame(path)
    base_roi = preprocess(base_frame[startY:finishY, startX:finishX])

    # Compute the base output path to use when saving frames
    category = path.split(os.sep)[1]
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
        
        roi = frame[startY:finishY, startX:finishX]
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