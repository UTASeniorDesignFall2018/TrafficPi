import cv2
import sys
import numpy as np
import time


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
        
        
        self.capture_video = capture_video # True - get video from webcam, False - get from file

        # config stuff for saving segmented video files
        self.config_save_frames = False # True - segment video from webcam or file
        self.threshold = threshold # the threshold to begin saving (could be better with connected components)
        self.fps = fps # frame rate to save at
        self.past_seconds_saved = past_seconds_saved 
        
        # Data stuff
        self._past_frames = Queue(maxsize=self.past_seconds_saved * self.fps)
        self._saving_frames = False
        self._total_pixels = 0

        # Sometimes delaying saving the file is helpful and it can reduce the number of files we have
        self.delay_saving = True
        self.delay_frames = delay_frames
        self._remaining_frames = 0
        

        # Option to enable live video capture
        if self.capture_video:
            try:
                self.cap = cv2.VideoCapture(0)
            except:
                print("No webcam or something")
                quit()
        else:
            if video_file == None:
                print("Boy, add the video file")
                quit()
            self.cap = cv2.VideoCapture(video_file)
        
        if self.config_save_frames:
            self.file_name = time.strftime('%d_%b_%Y %H_%M_%S.avi', time.localtime())
            self.out = cv2.VideoWriter(self.file_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080)) # should probably change this to 720 to save space
        
    def start_processing(self):

        self.setup_processing()

        # Loop over each frame
        while(True):
            ret, frame = self.cap.read()
            if not ret:
                break

            start_time = time.time()

            self.current_frame = frame

            new_frame = frame

            #new_frame = self.process_frame(frame)
            new_frame = self.process_frame_better_fps(frame)

            finish_processing = time.time()

            print("Process time:", finish_processing - start_time, "FPS:", 1 / (finish_processing - start_time))

            if self.config_save_frames:
                if (cv2.countNonZero(new_frame) / self._total_pixels > self.threshold or (self.delay_saving and self._remaining_frames > 0)):
                    self.save_frames()

                    self._remaining_frames -= 1
                else:
                    self._saving_frames = False
                    if self.out.isOpened():
                        self.out.release()
                    self._past_frames.put(frame)
            
            cv2.imshow('frame', new_frame)
            
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break

    def setup_processing(self):

        # Initialize variables used in processing
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
        self.bg_subtractor_fps = cv2.bgsegm.createBackgroundSubtractorCNT()
        # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

        # Get total pixel count
        ret, frame = self.cap.read()
        self._total_pixels = frame.size / 3

        # Ignore first 90 frames
        i = 1
        while(i < 200):
            i += 1
            self.cap.read()

    def process_frame(self, frame):
        new_frame = cv2.resize(frame, (640, 480))

        self._total_pixels = new_frame.size / 3

        new_frame = cv2.blur(new_frame, (2, 2))
        new_frame = self.bg_subtractor.apply(new_frame)

        # new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_OPEN, self.morph_kernel)

        new_frame = self.conncted_components(new_frame, threshold=1000)

        return new_frame

    def process_frame_better_fps(self, frame):
        new_frame = cv2.resize(frame, (640, 480))

        self._total_pixels = new_frame.size / 3

        new_frame = cv2.blur(new_frame, (5, 5))
        new_frame = self.bg_subtractor_fps.apply(new_frame)

        #new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_OPEN, np.ones((5,5)))
        #new_frame = cv2.dilate(new_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        new_frame = cv2.dilate(new_frame, np.ones((7,7)))

        new_frame = self.conncted_components(new_frame, threshold=1000)

        new_frame = self.fill_holes(new_frame)

        return new_frame

    def save_frames(self):
        # If we are still saving files, dont make a new one for each frame
        # just keep writing to the same file.

        if self._saving_frames:
            self.out.write(self.current_frame)
        else:
            self.file_name = time.strftime('%d_%b_%Y %H_%M_%S.avi', time.localtime())

            self.out.open(self.file_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

            self._saving_frames = True

            self._remaining_frames = self.delay_frames

            while True:
                fr = self._past_frames.get()
                if fr is None:
                    break
                self.out.write(fr)

    def __del__(self):
        self.cap.release()
        if self.config_save_frames:
            self.out.release()
        cv2.destroyAllWindows()

    def conncted_components(self, frame, threshold=1000):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=4)
        sizes = stats[:, -1]

        if len(sizes) == 1:
            return frame
            
        out = np.zeros(frame.shape, dtype='uint8')
        for l in range(1, len(sizes)):
            if sizes[l] > threshold:
                out[output == l] = 255

        return out

    def fill_holes(self, frame):
        im_floodfill = frame.copy()
        
        h, w = frame.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        return frame | im_floodfill_inv

pp = Preprocessor(capture_video=False, video_file=sys.argv[1], threshold=0.03)
pp.start_processing()
