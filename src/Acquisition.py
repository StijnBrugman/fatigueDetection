import cv2, dlib
import numpy as np
import time, os, glob, threading
from imutils import face_utils

from src.Settings import ABS_PATH


class Acquisition(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True

        #Initializing the camera and taking the instance
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FPS, 40)

        #Initializing the face detector and landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.eye_landmarks = {'left'  : [], 'right' : []}

        (self.start_l, self.end_l) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.start_r, self.end_r) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # nose, chin, l_eye, r_eye, l_mouth, r_mouth
        self.index_orientation_landmarks = [30, 8, 36, 45, 48, 54]

        self.start_time = time.time()
        self.counter = 0    
        self.buffer = []
        self.accesible = False
        self.safe = False

        self.old_frames = {}
        self.frames = {}
        self.frame_indx_len = 0
        self.frame = None
    
    def run(self):
        print("[INFO] Acquisition Thread Opened")

        files = glob.glob('/Users/stijnbrugman/PycharmProjects/fatigueDetection/frames/*')
        for f in files:
            os.remove(f)
        
        print("[INFO] All old frames have been removed")

        timer = time.time()
        FPS_array = np.array([])

        while self.running:
            start_time = time.time()
            _, frame = self.camera.read()
            
            

            # Compression
            # frame = self.compress_image(frame, .3)

            # Cropping
            x_lim_frame = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
            y_lim_frame = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

            # print(width, height)
            frame = frame[0:y_lim_frame + 100, x_lim_frame - 200:x_lim_frame + 200]
            # frame = frame[0:height/2, width/2 - 50: width/2 + 50]
            # cv2.imwrite("/Users/stijnbrugman/PycharmProjects/fatigueDetection/frames/test.png", frame)
            # print(frame)
            

            # GrayScaling
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray_frame, 0)
            timestamp = time.time() - self.start_time
            time_key = str("{:.2f}".format(timestamp))
            self.frames[time_key] = frame
            
            if len(faces) < 1:
                self.counter += 1
                if self.counter >= 500:
                    self.counter = 0
                    print("[WARNING] No face is detected")
            else:
                self.counter = 0            

            # if multiple faces are detected get closesed
            if len(faces) > 1: faces = self.get_closesed_face(faces)

            for face in faces:
                
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = self.predictor(gray_frame, face)
                
                landmarks = face_utils.shape_to_np(landmarks)
                self.eye_landmarks['left'] = landmarks[self.start_l:self.end_l]
                self.eye_landmarks['right'] = landmarks[self.start_r:self.end_r]

                # Head orientation
                orientation_landmarks = [
                    landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54], 
                ]

                self.add_to_buffer({
                    'eye':(timestamp, self.eye_landmarks),
                    'orientation': orientation_landmarks,
                    'img': frame
                })

                for landmark in orientation_landmarks:
                    x,y = landmark
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

                for landmark_l, landmark_r in zip(self.eye_landmarks['left'], self.eye_landmarks['right']):
                    (x1, y1) = landmark_l
                    (x2, y2) = landmark_r
                    cv2.circle(frame, (x1, y1), 2, (255, 255, 255), -1)
                    cv2.circle(frame, (x2, y2), 2, (255, 255, 255), -1)
            self.set_frame(frame)
            self.accesible = True
                # self.buffer.append(self.eye_landmarks)
            # cv2.imshow("Frame", frame)
            
            if time.time() - timer > 20: 
                timer = time.time()
                
                FPS = 1 / (time.time() - start_time)
                FPS_array = np.append(FPS_array, FPS)
                FPS_array = FPS_array[-50:]
                print("[INFO] Framerate Acquisition-Threat is: {}".format(np.average(FPS_array)))
            
            #print("[INFO] Framerate Acquistion-Thread is: {}".format(FPS))
        self.camera.release()
        print("[INFO] Acquisition Thread Closed")
        
    def get_closesed_face(self, faces):
        index, size = 0, 0
        for i, face in enumerate(faces): 
            length = face.right() - face.left()
            if length > size: index, size = i, length
        return [faces[index]]
    
    def frame_accisible(self):
        return self.accesible

    def set_frame(self, frame):
        self.frame = frame
    
    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False

    def get_from_buffer(self):
        return self.buffer.pop(0)
    
    def add_to_buffer(self, element):
        self.buffer.append(element)

    def buffer_availble(self):
        return self.buffer
    
    def set_setting(self, safe):
        self.safe = safe

    def safe_frames(self, frames_index):
        if not self.safe: return
        for time in frames_index:
            index = str("{:.2f}".format(time))
            file_name = ABS_PATH + "/frames/frame[{}].png".format(index)
            frame = self.frames.get(index)
            if frame is None: continue
            cv2.imwrite(file_name, frame)
        print("[INFO] Frames have been saved")


    @staticmethod
    def compress_image(frame, percentage = .75):
        width = int(frame.shape[1] * percentage)
        height = int(frame.shape[0] * percentage)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    def remove_non_blink_frames(self, frames_index):
        # print(self.frames.keys(), frames_index)
        temp_frames = self.old_frames.copy()

        if len(frames_index) == self.frame_indx_len: return
        len_index = len(frames_index)
        frames_index = frames_index[self.frame_indx_len:]
        self.frame_indx_len = len_index

        
        for time in frames_index:
            index = str("{:.2f}".format(time))
            temp_frames[index] = self.frames.get(index)
        
        self.frames = temp_frames.copy()
        self.old_frames = temp_frames.copy()
        
        return self.frames
        # print(self.frames.keys())

