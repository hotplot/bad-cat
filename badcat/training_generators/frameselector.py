import cv2, os

class FrameSelector:
    def __init__(self, path, roi_coords):
        self.path = path
        self.roi_coords = roi_coords
        self.frame_index = None
        self.frame = None
        self.video_cap = None
        self.window_name = os.path.split(path)[1]


    def __trackbar_change(self, value):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.frame_index = value
        _, self.frame = self.video_cap.read()
        if self.frame is not None:
            image = self.frame.copy()
            image = cv2.rectangle(
                image,
                (self.roi_coords[0], self.roi_coords[1]),
                (self.roi_coords[2], self.roi_coords[3]),
                (0,255,255),
                2
            )
            cv2.imshow(self.window_name, image)


    def select_frame(self):
        self.video_cap = cv2.VideoCapture(self.path)
        frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('position', self.window_name, 0, frame_count - 1, self.__trackbar_change)
        self.__trackbar_change(0)

        if cv2.waitKey() == ord('a'):
            # If the user presses 'a', treat it as 'no frame selected'
            self.frame_index = -1
            self.frame = None
        
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)  # otherwise the window doesn't close...

        return self.frame_index, self.frame
