import cv2
import sys
import numpy as np
import time

# TODO: save frames fix
class Queue:
    def __init__(self, maxsize=0):
        self.size = 0
        self.items = []
        self.maxsize = maxsize

    def get(self):
        if len(self.items) == 0:
            return None
        return self.items.pop()

    def put(self, item):
        if len(self.items) == self.maxsize:
            self.items.pop()
            self.items.insert(0, item)
        else:
            self.items.insert(0, item)
        
        self.size += 1


class Preprocessor:

    def __init__(self, capture_video=True, threshold=0.1, fps=30,
                 past_seconds_saved=1, video_file=None, delay_frames=60):
        self.capture_video = capture_video
        self.threshold = threshold
        self.fps = fps
        self.past_seconds_saved = past_seconds_saved

        self.past_frames = Queue(maxsize=self.past_seconds_saved * self.fps)
        self.saving_frames = False

        self.delay_saving = True
        self.delay_frames = delay_frames
        self.remaining_frames = 0

        self.total_pixels = 0

        # Option to enable live video capture
        if self.capture_video:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(video_file)
        
        self.file_name = time.strftime('%d_%b_%Y %H_%M_%S.avi', time.localtime())
        self.out = cv2.VideoWriter(self.file_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))
        
    def start_processing(self):

        self.setup_processing()

        # Loop over each frame
        while(True):
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame = frame

            new_frame = self.process_frame(frame)

            if cv2.countNonZero(new_frame) / self.total_pixels > self.threshold or (self.delay_saving and self.remaining_frames > 0):
                self.save_frames()

                self.remaining_frames -= 1
            else:
                self.saving_frames = False
                if self.out.isOpened():
                    self.out.release()
                self.past_frames.put(frame)
            
            cv2.imshow('frame', new_frame)
            
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break

    def setup_processing(self):

        # Initialize variables used in processing
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
        # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

        # Get total pixel count
        ret, frame = self.cap.read()
        self.total_pixels = frame.size / 3

        # Ignore first 90 frames
        i = 1
        while(i < 200):
            i += 1
            self.cap.read()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))

        self.total_pixels = frame.size / 3

        # new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_frame = cv2.blur(frame, (10, 10))
        new_frame = self.bg_subtractor.apply(new_frame)
        
        # new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_OPEN, self.morph_kernel)

        return new_frame

    def save_frames(self):
        # If we are still saving files, dont make a new one for each frame
        # just keep writing to the same file.

        if self.saving_frames:
            self.out.write(self.current_frame)
        else:
            self.file_name = time.strftime('%d_%b_%Y %H_%M_%S.avi', time.localtime())

            self.out.open(self.file_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

            self.saving_frames = True

            self.remaining_frames = self.delay_frames

            while True:
                fr = self.past_frames.get()
                if fr is None:
                    break
                self.out.write(fr)

    def __del__(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


pp = Preprocessor(capture_video=False, video_file=sys.argv[1], threshold=0.03)
pp.start_processing()