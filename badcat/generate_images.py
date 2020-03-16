import argparse, glob, os.path, pickle

import cv2
import numpy as np

from .utils import extract_roi, preprocess_roi, extract_hull, display_preview


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


def save_image(path, image):
    folder = os.path.split(path)[0]
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, image)


def process_video(path, output_dir, avg_roi, roi_coords, moving_thresh, still_thresh, show_preview):
    # Compute the base output path to use when saving frames
    category = path.split(os.sep)[-2]
    filename = os.path.splitext(os.path.basename(path))[0]

    # Pre-process the averaged ROI
    avg_roi = cv2.resize(avg_roi, (224, 224))
    avg_roi = cv2.cvtColor(avg_roi, cv2.COLOR_BGR2GRAY)
    avg_roi = cv2.GaussianBlur(avg_roi, (13, 13), 0)
    avg_roi = cv2.equalizeHist(avg_roi)

    # Open the video and process each frame
    index = 0

    num_occupied_frames = 0
    empty_frames = []
    
    vc = cv2.VideoCapture(path)
    while vc.isOpened():
        # Read the next frame
        ret, frame = vc.read()
        if ret is False or frame is None:
            break
        
        # Preprocess the frame
        curr_roi = extract_roi(frame, roi_coords)
        curr_roi, processed_curr_roi = preprocess_roi(curr_roi)

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

        # Book keeping
        index += 1
    
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

    vc.release()


def get_avg_roi(roi_coords, path):
    x1, y1, x2, y2 = roi_coords
    
    # Create a np array the same size as the ROI to use as an accumulator
    avg = np.zeros((y2 - y1, x2 - x1, 3), dtype=float)
    count = 0

    # Sum each of the frames into the array
    vc = cv2.VideoCapture(path)
    while vc.isOpened():
        ret, frame = vc.read()
        if ret is False or frame is None:
            break

        roi = frame[y1:y2, x1:x2]
        avg += roi
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
            args['preview']
        )

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
