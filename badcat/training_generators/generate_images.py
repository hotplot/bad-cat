import argparse, glob, os.path, pickle

import cv2
import numpy as np

from .frameselector import FrameSelector
from ..utils import preprocess, extract_hull, display_preview


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-i', '--input', default='train_movies')
    ap.add_argument('-o', '--output', default='train_images')
    ap.add_argument('-m', '--moving_threshold', type=float, default=0.15)
    ap.add_argument('-s', '--still_threshold', type=float, default=0.025)
    ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
    ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')
    ap.add_argument('--preview', action='store_true', help='show a preview of extracted data')    
    return vars(ap.parse_args())


def get_reference_frame(path, ref_frame_indexes, roi_coords):
    # If no reference frame is known, prompt the user to select one
    if not path in ref_frame_indexes:
        selector = FrameSelector(path, roi_coords)
        index, frame = selector.select_frame()

        # Store the reference frame index
        ref_frame_indexes[path] = index
        with open('ref_frame_indexes.pickle', 'wb') as f:
            pickle.dump(ref_frame_indexes, f)
    
    # Load and return the reference frame for this video
    # An index of -1 indicates that all frames contain the object we want to detect.
    if path in ref_frame_indexes:
        if ref_frame_indexes[path] == -1:
            return None
        else:
            vc = cv2.VideoCapture(path)
            vc.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_indexes[path])
            _, frame = vc.read()
            return frame


def save_image(path, image):
    folder = os.path.split(path)[0]
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, image)


def process_video(path, output_dir, ref_roi, roi_coords, moving_thresh, still_thresh, show_preview):
    # Compute the base output path to use when saving frames
    category = path.split(os.sep)[-2]
    filename = os.path.splitext(os.path.basename(path))[0]

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
        
        # Preprocess the frame
        curr_roi, processed_curr_roi = preprocess(frame, roi_coords)

        # Form a complex hull around the movement in the frame,
        # and choose a category based on the magnitude of the movement.
        # If the reference ROI is None, all frames are assumed to contain movement.
        hull, hull_proportion = (None, 0)
        if ref_roi is not None:
            hull, hull_proportion = extract_hull(processed_curr_roi, ref_roi)

        # Check how much movement occurs in this frame, and save the ROI
        if ref_roi is None or hull_proportion > moving_thresh:
            output_path = os.path.join(output_dir, category, f'{filename}-{index}.jpg')
            save_image(output_path, curr_roi)
            num_moving_frames += 1
        elif hull_proportion < still_thresh:
            stationary_frames.append(curr_roi)

        if show_preview:
            display_preview(curr_roi, hull, hull_proportion)

        # Book keeping
        index += 1
    
    # Save the frames where no movement occurred, but only enough to equal the number of 
    # frames where movement *did* occur.
    # Normally there are far more still frames than moving frames, and saving them all 
    # results in significantly unbalanced class sizes.
    num_saved = 0
    while num_saved < num_moving_frames and num_saved < len(stationary_frames):
        output_path = os.path.join(output_dir, 'none', f'{filename}-{index}.jpg')
        save_image(output_path, stationary_frames[num_saved])
        num_saved += 1
        index += 1

    vc.release()


def main():
    args = parse_args()

    # Compute the coordinates of the region of interest
    roi_coords = (
        args['roi_x'],
        args['roi_y'],
        args['roi_x'] + args['roi_width'],
        args['roi_y'] + args['roi_height']
    )

    # Load pickled 'reference frame' info
    ref_frame_indexes = {}
    try:
        with open('ref_frame_indexes.pickle', 'rb') as f:
            ref_frame_indexes = pickle.load(f)
    except:
        pass

    # Loop over and process each training video
    paths = sorted(glob.glob(os.path.join(args['input'], '**/*.mp4')))
    for path in paths:
        ref_frame = get_reference_frame(path, ref_frame_indexes, roi_coords)
        if ref_frame is not None:
            _, ref_roi = preprocess(ref_frame, roi_coords)
        else:
            ref_roi = None
        
        process_video(
            path,
            args['output'],
            ref_roi,
            roi_coords,
            args['moving_threshold'],
            args['still_threshold'],
            args['preview']
        )


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()