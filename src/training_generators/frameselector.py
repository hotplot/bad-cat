import cv2, os

class FrameSelector:
    def __init__(self, path):
        self.path = path
        self.frame_index = None
        self.frame = None
        self.video_cap = None
        self.window_name = os.path.split(path)[1]


    def __trackbar_change(self, value):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.frame_index = value
        _, self.frame = self.video_cap.read()
        if self.frame is not None:
            cv2.imshow(self.window_name, self.frame)


    def select_frame(self):
        self.video_cap = cv2.VideoCapture(self.path)
        frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('position', self.window_name, 0, frame_count - 1, self.__trackbar_change)
        self.__trackbar_change(0)
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)  # otherwise the window doesn't close...

        return self.frame_index, self.frame
