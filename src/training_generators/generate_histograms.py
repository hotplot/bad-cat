import argparse, glob, os.path, pickle

import cv2
import numpy as np
import pandas as pd

from ..utils import preprocess, extract_hull, extract_histograms, display_preview


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-i', '--input', default='train_movies')
    ap.add_argument('-o', '--output', default='histograms.pickle')
    ap.add_argument('--nth', type=int, default=5, help='only consider every Nth frame')
    ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
    ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')
    ap.add_argument('--preview', action='store_true', help='show a preview of extracted data')    
    return vars(ap.parse_args())


def process_video(path, roi_coords, nth_frame, show_preview):
    # Determine the cat class this video represents
    category = path.split(os.sep)[-2]

    # Open the video and process each frame
    index = 0
    processed_prev_roi = None

    histograms_df = pd.DataFrame(columns=['label', 'hull_proportion', 'fg_hist', 'bg_hist'])

    vc = cv2.VideoCapture(path)
    while vc.isOpened():
        # Read the current frame and crop out the ROI
        ret, frame = vc.read()
        if ret is False or frame is None:
            break

        if processed_prev_roi is None:
            _, processed_prev_roi = preprocess(frame, roi_coords)
            continue
        
        if index % nth_frame != 0:
            index += 1
            continue
        
        # Preprocess the frame (grayscale, blur, equalise hist)
        curr_roi, processed_curr_roi = preprocess(frame, roi_coords)

        # Form a complex hull around the movement in the frame,
        # and choose a category based on the magnitude of the movement
        hull, hull_proportion = extract_hull(processed_curr_roi, processed_prev_roi)

        hist_label = category
        if hull_proportion < 0.05:
            hist_label = 'none'
        elif hull_proportion > 0.5:
            hist_label = 'other'

        # Extract the fore and background histograms and store training data
        fg_hist, bg_hist = extract_histograms(curr_roi, hull)
        histograms_df = histograms_df.append({
            'label': hist_label,
            'hull_proportion': hull_proportion,
            'fg_hist': fg_hist,
            'bg_hist': bg_hist,
        }, ignore_index=True)

        # Book keeping
        index += 1
        processed_prev_roi = processed_curr_roi

        if show_preview:
            display_preview(curr_roi, hull, hull_proportion)

    vc.release()

    trunc_path = os.sep.join(path.split(os.sep)[-2:])
    print(f'{trunc_path}: {len(histograms_df)} histograms')

    return histograms_df


def main():
    args = parse_args()

    # Compute the coordinates of the region of interest
    roi_coords = (
        args['roi_x'],
        args['roi_y'],
        args['roi_x'] + args['roi_width'],
        args['roi_y'] + args['roi_height']
    )

    # Loop over and process each training video
    histograms_df = pd.DataFrame()

    paths = sorted(glob.glob(os.path.join(args['input'], '**/*.mp4')))
    for path in paths:
        video_df = process_video(path, roi_coords, args['nth'], args['preview'])
        histograms_df = histograms_df.append(video_df, ignore_index=True)
    
    histograms_df.to_pickle(args['output'])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
