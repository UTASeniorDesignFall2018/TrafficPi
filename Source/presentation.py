import cv2
import sys
import numpy as np
import time
import math

from statistics import median

from car_info import RegressionModel


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


class SimilarTriangles:
    def __init__(self):
        # Hard-coded calibration, should be learned.
        # [distance to, width]
        # self.calibration = [23, 13.666667]
        # self.calibration = [0.125, 0.25]
        self.calibration = [94.66, 88]
    
    def get_road_length(self, distance):
        return distance * self.calibration[1] / self.calibration[0]


class Car:
    def __init__(self, car, id=0):
        self.id = id
        self.car = car
        self.velocity = 0
        
        self.first_seen = 0
        self.last_seen = time.time()

        self.mids = [((car[0] + car[1])/2, (car[2] + car[3])/2)]
        self.next_pos = self.mids[0]

        self.car_width = [car[1] - car[0]]
        self.car_height = [car[3] - car[2]]

        self.velocities = []

        self.num_frames = 1

    def update(self, car):
        self.car = car
        mid = ((car[0] + car[1])/2, (car[2] + car[3])/2)

        self.velocity = (np.array(mid) - np.array(self.mids[-1])) / (self.last_seen - time.time())
        self.velocities.append(self.velocity[0])

        self.next_pos = mid + (self.last_seen - time.time()) * self.velocity

        if self.first_seen == 0:
            self.first_seen = time.time()

        self.last_seen = time.time()
        self.mids.append(mid)

        self.car_width.append(car[1] - car[0])
        self.car_height.append(car[3] - car[2])

        self.num_frames += 1
    
    def get_next_pos(self):
        x, y = self.next_pos
        w = median(self.car_width)
        h = median(self.car_height)
        next_car = [x - w/2, x + w/2, y - h/2, y + h/2]
        next_car = [round(n) for n in next_car]
        return next_car

    def get_pos(self):
        x, y = self.mids[-1]
        w = median(self.car_width[math.floor(len(self.car_width)/2):])
        h = median(self.car_height)
        next_car = [x - w/2, x + w/2, y - h/2, y + h/2]
        next_car = [round(n) for n in next_car]
        return next_car


class CarTracker:
    def __init__(self):
        self.cars = []
        self.rm = RegressionModel()
        self.st = SimilarTriangles()
    
    def IOU(self, car1, car2):
        #car1 ~ (left, right, top, bottom)
        #car2 ~ same ^

        area_c1 = (car1[1] - car1[0]) * (car1[3] - car1[2])
        area_c2 = (car2[1] - car2[0]) * (car2[3] - car2[2])

        inter_w = min(car1[3], car2[3]) - max(car1[2], car2[2])
        inter_h = min(car1[1], car2[1]) - max(car1[0], car2[0])

        if inter_w <= 0 or inter_h <= 0:
            return 0

        intersection = inter_w * inter_h
        return intersection / (area_c1 + area_c2 - intersection)

    def update_pos(self):
        t = time.time()
        for car in self.cars:
            #print(car.mids, car.mids[-1])
            car.next_pos = np.array(car.mids[-1]) + (car.last_seen - time.time()) * car.velocity

            if car.next_pos[0] < 0 or car.next_pos[0] > 640:
                self.cars.remove(car)
                avg = np.mean(car.velocities)

                if car.last_seen - car.first_seen < 0.25:
                    continue

                print("Car stats:")

                dist_to_car = self.rm.get_distance(car.car_width[math.floor(len(car.car_width)/2)], car.car_height[math.floor(len(car.car_width)/2)])
                road_len = self.st.get_road_length(dist_to_car)
                print("Distance to car:", int(dist_to_car), "ft")
                print("Road len:", int(road_len), "ft")
                print("Time:", round(car.num_frames / 30, 2), "seconds")
                print("Speed:", round((road_len / (car.num_frames/30) / 1.467)[0,0], 2), "mph")

            if t - car.last_seen > 1:
                self.cars.remove(car)

    def update_car(self, car1):
        score, car = self.find_car(car1)

        if car == None or score < 0.4:
            self.cars.append(Car(car1))
            return

        car.update(car1)

    def find_car(self, car1):
        best_match = (-1, None)
        for car in self.cars:
            iou = self.IOU(car.get_next_pos(), car1)
            if iou > best_match[0]:
                best_match = (iou, car)
        
        return best_match


class Preprocessor:
    def __init__(self, capture_video=True, threshold=0, fps=30,
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

        # Car tracking
        self.car_tracker = CarTracker()
        self.cars_in_frame = []
        self.cars_in_previous = []
        

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
    
    def skip_frames(self, n):
        for _ in range(n):
            ret, frame = self.cap.read()

            if not ret:
                break
            
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break

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

            new_frame = self.process_frame(frame)
            #new_frame = self.process_frame_better_fps(frame)

            finish_processing = time.time()

            #print("Process time:", finish_processing - start_time, "FPS:", 1 / (0.000001 + finish_processing - start_time))

            # if self.config_save_frames:
            #     if (cv2.countNonZero(new_frame) / self._total_pixels > self.threshold or (self.delay_saving and self._remaining_frames > 0)):
            #         self.save_frames()

            #         self._remaining_frames -= 1
            #     else:
            #         self._saving_frames = False
            #         if self.out.isOpened():
            #             self.out.release()
            #         self._past_frames.put(frame)
            
            # self.current_frame = new_frame
            # self.save_frames()

            cv2.imshow('frame', new_frame)
            
            key = cv2.waitKey(int(33 - (finish_processing - start_time)))
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

        self.boxes = []
        self.all_boxes = []

        # Ignore first 90 frames
        #self.skip_frames(200)

    def process_frame(self, frame):
        old_frame = cv2.resize(frame, (640, 480))
        new_frame = old_frame

        self._total_pixels = new_frame.size / 3

        new_frame = cv2.blur(new_frame, (5,5))
        new_frame = self.bg_subtractor.apply(new_frame)

        # new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_OPEN, self.morph_kernel)

        new_frame = self.conncted_components(new_frame, threshold=1000)

        # Update each car's position and add new ones
        for car in self.cars_in_frame:
            self.car_tracker.update_car(car)
        
        
        self.car_tracker.update_pos()

        # Draw box for each car on screen, also draw a line to where their next pos should be
        for car in self.car_tracker.cars:
            pos = car.get_pos()
            old_frame = cv2.rectangle(old_frame, (pos[0], pos[2]), (pos[1], pos[3]), (255, 255, 255), 2)
            mp = car.mids[-1]
            next_pos = car.next_pos
            old_frame = cv2.line(old_frame, (int(mp[0]), int(mp[1])), (int(next_pos[0]), int(next_pos[1])), (0, 255, 0), 4)

        return old_frame

    def process_frame_better_fps(self, frame):
        new_frame = cv2.resize(frame, (640, 480))

        self._total_pixels = new_frame.size / 3
        
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        
        new_frame = cv2.blur(new_frame, (3,3))
        
        new_frame = self.bg_subtractor_fps.apply(new_frame)
        
        new_frame = cv2.dilate(new_frame, np.ones((5, 5)))

        new_frame = self.conncted_components(new_frame, threshold=3000)

        # for car in self.cars_in_frame:
        #     new_frame = cv2.rectangle(new_frame, (car[0], car[2]), (car[1], car[3]), (255, 255, 255), 2)

        # Update each car's position and add new ones
        for car in self.cars_in_frame:
            self.car_tracker.update_car(car)
        
        self.car_tracker.update_pos()

        # Draw box for each car on screen, also draw a line to where their next pos should be
        for car in self.car_tracker.cars:
            pos = car.get_pos()
            new_frame = cv2.rectangle(new_frame, (pos[0], pos[2]), (pos[1], pos[3]), (255, 255, 255), 2)
            mp = car.mids[-1]
            next_pos = car.next_pos
            new_frame = cv2.line(new_frame, (int(mp[0]), int(mp[1])), (int(next_pos[0]), int(next_pos[1])), (30, 30, 30), 2)

        return new_frame

    def save_frames(self, frame=None):
        # If we are still saving files, dont make a new one for each frame
        # just keep writing to the same file.

        if self._saving_frames:
            self.out.write(frame)
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

    def conncted_components(self, frame, threshold=3000):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=4)
        sizes = stats[:, -1]

        if len(sizes) == 1:
            return frame
            
        out = np.zeros(frame.shape, dtype='uint8')

        # self.boxes = []
        self.cars_in_previous = self.cars_in_frame
        self.cars_in_frame = []

        for l in range(1, len(sizes)):
            
            if sizes[l] > threshold:
                out[output == l] = 255
                
                box = np.zeros(frame.shape, dtype='uint8')
                box[output == l] = 255
                x,y,w,h = cv2.boundingRect(box)

                car = [x, x+w, y, y+h]
                self.cars_in_frame.append(car)

                # self.boxes.append((x,y,w,h))
                # self.all_boxes.append((x,y,w,h))

        return out

    def fill_holes(self, frame):
        im_floodfill = frame.copy()
        
        h, w = frame.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        return frame | im_floodfill_inv




while True:
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    pp = Preprocessor(capture_video=False, video_file=sys.argv[1], threshold=0.03)
    pp.start_processing()

