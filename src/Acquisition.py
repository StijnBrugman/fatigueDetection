
import threading
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import os
import glob


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
        self.eye_landmarks = {
            'left'  : [],
            'right' : []
        }

        (self.start_l, self.end_l) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.start_r, self.end_r) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.buffer = []

        self.start_time = time.time()

        self.frame = None

        self.accesible = False
        self.frames = {}

        self.safe = False
    


    def run(self):
        print("[INFO] Acquisition Thread Opened")

        files = glob.glob('/Users/stijnbrugman/PycharmProjects/fatigueDetection/frames/*')
        for f in files:
            os.remove(f)
        
        print("[INFO] All old frames have been removed")

        self.timer = time.time()
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
            
            # print("frames",len(self.frames))

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
                self.add_to_buffer((timestamp, self.eye_landmarks))

            for landmark_l, landmark_r in zip(self.eye_landmarks['left'], self.eye_landmarks['right']):
                (x1, y1) = landmark_l
                (x2, y2) = landmark_r
                cv2.circle(self.frame, (x1, y1), 1, (255, 255, 255), -1)
                cv2.circle(self.frame, (x2, y2), 1, (255, 255, 255), -1)
            self.set_frame(frame)
            self.accesible = True
                # self.buffer.append(self.eye_landmarks)
            # cv2.imshow("Frame", frame)
            
            if time.time() - self.timer > 20: 
                self.timer = time.time()
                
                FPS = 1 / (time.time() - start_time)
                FPS_array = np.append(FPS_array, FPS)
                FPS_array = FPS_array[-50:]
                print("[INFO] Framerate Acquisition-Threat is: {}".format(np.average(FPS_array)))
            
            #print("[INFO] Framerate Acquistion-Thread is: {}".format(FPS))
        self.camera.release()
        print("[INFO] Acquisition Thread Closed")
        
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
            file_name = "C:/Users/JohnBrugman/fatigueDetection/frames/" + "frame[{}].png".format(index)
            cv2.imwrite(file_name, self.frames.get(index))
        print("[INFO] Frames have been saved")


    @staticmethod
    def compress_image(frame, percentage = .75):
        width = int(frame.shape[1] * percentage)
        height = int(frame.shape[0] * percentage)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    

    