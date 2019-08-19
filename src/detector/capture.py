import datetime, time
import logging, picamera, traceback
import numpy as np

from PIL import Image


class FrameSink:
    """A custom file-like output which accepts raw RGB data from picamera
    and forwards it to the classifier as a numpy array."""

    def __init__(self, roi, capture_size, target_size, classification_interval, queue):
        self.crop_coords = (
            roi['x'],
            roi['y'],
            roi['x'] + roi['w'],
            roi['y'] + roi['h']
        )

        self.capture_size = capture_size
        self.target_size = target_size
        self.queue = queue

        self.classification_interval = classification_interval
        self.next_classification_time = time.time()
    
    def write(self, data):
        now = time.time()
        if now > self.next_classification_time:
            self.next_classification_time += self.classification_interval
            image = Image.frombytes('RGB', self.capture_size, data)
            image = image.crop(self.crop_coords)
            image = image.resize(self.target_size)
            self.queue.put(np.array(image))


def capture_frames(roi, capture_size, target_size, classification_interval, frame_queue, should_stop, cat_detected):
    """Starts streaming from the RPi camera and forwards cropped, resized frames
    to the classifier via `frame_queue` until `should_stop` is set."""
    
    with picamera.PiCamera() as camera:
        camera.resolution = capture_size
        camera.framerate = 20

        # Begin recording to the circular buffer (used for saving footage of events)
        buffer = picamera.PiCameraCircularIO(camera, seconds=20)
        camera.start_recording(buffer, format='h264', splitter_port=1)

        # Begin recording to the custom output (used for sending frames to the classifier)
        frame_sink = FrameSink(roi, capture_size, target_size, classification_interval, frame_queue)
        camera.start_recording(frame_sink, format='rgb', splitter_port=2)

        # Wait until we are asked to exit, and save footage of any cat activity
        try:
            while not should_stop.is_set():
                cat_detected.wait(timeout=0.1)
                if cat_detected.is_set():
                    cat_detected.clear()    # Bit of a hack, just to avoid long recordings if a cat stays on the doorstep
                    camera.wait_recording(10)
                    filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.h264')
                    buffer.copy_to(filename)
        finally:
            camera.stop_recording(splitter_port=2)
            camera.stop_recording(splitter_port=1)
            frame_queue.cancel_join_thread()
