import logging, traceback
import cv2


def stream_frames(url, frame_queue, should_stop):
    vc = cv2.VideoCapture(url)
    
    while not should_stop.is_set():
        # Attempt to read the next frame from the stream, and reconnect if
        # there was an error
        try:
            result, frame = vc.read()
        except:
            logging.warning('Reading stream failed')
            traceback.print_exc()
            vc.release()
            vc = cv2.VideoCapture(url)
            continue
        
        # Send the frame to the classifier and display it on screen
        if frame is not None:
            frame_queue.put(frame)
            cv2.imshow('Stream', frame)

        # Check for user quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_stop.set()

    # Tidy up resources
    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    frame_queue.cancel_join_thread()
