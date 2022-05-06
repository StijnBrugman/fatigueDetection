
import threading
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time

class Acquisition(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True

        #Initializing the camera and taking the instance
        self.camera = cv2.VideoCapture(0)

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

    def run(self):
        print("[INFO] Connection Acquisition is established")
        
        while self.running:
            _, frame = self.camera.read()
            

            # Compression
            frame = self.compress_image(frame, .3)
            

            # GrayScaling
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray_frame, 0)
            
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)




                landmarks = self.predictor(gray_frame, face)
                timestamp = time.time() - self.start_time
                landmarks = face_utils.shape_to_np(landmarks)
                self.eye_landmarks['left'] = landmarks[self.start_l:self.end_l]
                self.eye_landmarks['right'] = landmarks[self.start_r:self.end_r]
                self.add_to_buffer((timestamp,self.eye_landmarks))

            for landmark_l, landmark_r in zip(self.eye_landmarks['left'], self.eye_landmarks['right']):
                (x1, y1) = landmark_l
                (x2, y2) = landmark_r
                cv2.circle(self.frame, (x1, y1), 1, (255, 255, 255), -1)
                cv2.circle(self.frame, (x2, y2), 1, (255, 255, 255), -1)
            self.set_frame(frame)
            self.accesible = True
                # self.buffer.append(self.eye_landmarks)
            # cv2.imshow("Frame", frame)
        self.camera.release()
        
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

    @staticmethod
    def compress_image(frame, percentage = .75):
        width = int(frame.shape[1] * percentage)
        height = int(frame.shape[0] * percentage)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    

    