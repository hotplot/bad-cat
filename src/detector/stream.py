import logging, traceback
import cv2


def display_frame(frame, roi_coords):
    x1, x2, y1, y2 = roi_coords
    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 3)
    cv2.imshow('Stream', frame)


def stream_frames(url, roi_coords, frame_queue, should_stop):
    vc = cv2.VideoCapture(url)
    
    while not should_stop.is_set():
        # Attempt to read the next frame from the stream, and reconnect if
        # there was an error
        reconnect = False
        frame = None

        try:
            _, frame = vc.read()
        except:
            traceback.print_exc()
            reconnect = True
        
        if reconnect or frame is None:
            logging.warning('Reading stream failed; reconnecting')
            vc.release()
            vc = cv2.VideoCapture(url)
        
        # Send the frame to the classifier and display it on screen
        if frame is not None:
            frame_queue.put(frame)
            display_frame(frame, roi_coords)

        # Check for user quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_stop.set()

    # Tidy up resources
    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    frame_queue.cancel_join_thread()
