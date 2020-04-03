import argparse, glob, os.path, pickle

import cv2
import numpy as np

from .utils import iter_frames, extract_hull


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-i', '--input', default='train_movies')
    ap.add_argument('-o', '--output', default='train_images')
    ap.add_argument('-m', '--moving_threshold', type=float, default=0.15)
    ap.add_argument('-s', '--still_threshold', type=float, default=0.025)
    ap.add_argument('--nth', type=int, default=5, help='only consider every Nth frame')
    ap.add_argument('--roi_x', type=int, default=160, help='the top-left X coordinate of the ROI')
    ap.add_argument('--roi_y', type=int, default=0, help='the top-left Y coordinate of the ROI')
    ap.add_argument('--roi_width', type=int, default=800, help='the width of the ROI')
    ap.add_argument('--roi_height', type=int, default=720, help='the height of the ROI')
    ap.add_argument('--preview', action='store_true', help='show a preview of extracted data')    
    return vars(ap.parse_args())


def save_image(path, image):
    folder = os.path.split(path)[0]
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, image)


def display_preview(roi, hull, hull_proportion):
    output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    output = cv2.putText(output, f'Hull proportion: {hull_proportion}', (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    if hull is not None:
        output = cv2.drawContours(output.copy(), [hull], -1, (0,0,255))
    cv2.imshow('Output', output)
    if cv2.waitKey(0) == ord('q'):
        exit()


def process_video(path, output_dir, avg_roi, roi_coords, moving_thresh, still_thresh, nth, show_preview):
    # Compute the base output path to use when saving frames
    category = path.split(os.sep)[-2]
    filename = os.path.splitext(os.path.basename(path))[0]

    # Open the video and process each frame
    num_occupied_frames = 0
    empty_frames = []
    
    for index, curr_roi, processed_curr_roi in iter_frames(path, roi_coords, nth=nth):
        # Form a complex hull around the movement in the frame
        hull, hull_proportion = extract_hull(processed_curr_roi, avg_roi)

        # Check how much movement occurs in this frame, and save the ROI
        if hull_proportion > moving_thresh:
            output_path = os.path.join(output_dir, category, f'{filename}-{index}.jpg')
            save_image(output_path, curr_roi)
            num_occupied_frames += 1
        elif hull_proportion < still_thresh:
            empty_frames.append(curr_roi)

        if show_preview:
            display_preview(curr_roi, hull, hull_proportion)
    
    # Save the frames where no movement occurred, but only enough to equal the number of 
    # frames where movement *did* occur.
    # Normally there are far more still frames than moving frames, and saving them all 
    # results in significantly unbalanced class sizes.
    num_saved = 0
    while num_saved < num_occupied_frames and num_saved < len(empty_frames):
        output_path = os.path.join(output_dir, 'none', f'{filename}-{index}.jpg')
        save_image(output_path, empty_frames[num_saved])
        num_saved += 1
        index += 1


def get_avg_roi(roi_coords, path):
    avg = None
    count = 0

    # Sum each of the frames into the array
    for _, _, processed_roi in iter_frames(path, roi_coords):
        if avg is None:
            avg = np.zeros_like(processed_roi, dtype=np.float)
        avg += processed_roi
        count += 1
    
    # Compute average and return
    avg = avg / count
    avg = avg.astype(np.uint8)

    return avg


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
    paths = sorted(glob.glob(os.path.join(args['input'], '**/*.mp4')))
    for path in paths:
        # Extract the average frame from each video, to use for motion detection
        avg_roi = get_avg_roi(roi_coords, path)

        if args['preview']:
            cv2.imshow(f'Average for {os.path.basename(path)}', avg_roi)
            if cv2.waitKey(0) == ord('q'):
                break
        
        # Process the video to extract empty and occupied frames
        process_video(
            path,
            args['output'],
            avg_roi,
            roi_coords,
            args['moving_threshold'],
            args['still_threshold'],
            args['nth'],
            args['preview']
        )

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
