#Importing OpenCV Library for basic image processing functions
import cv2

# Numpy for array related functions
import numpy as np

# Dlib for deep learning based Modules and face landmark detection
import dlib

#face_utils for basic operations of conversion
from imutils import face_utils

import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from visualization import Visualization

vis = Visualization()

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# def animate():
#     x_values.append(time.time())
#     y_values.append(EAR)
#
#     plt.cla()
#     plt.plot(x_values, y_values)


previous_time = 0
current_time = 0
FPS = 0
EAR = 0

# plt.figure(figsize=(3,3))
# figure, ax = plt.subplots(figsize=(2,2))
#
# plt.ion()
#
# x_values = []
# y_values = []
#
# plot_EAR, = ax.plot(x_values, y_values)



# def update_figure():
#     # figure.set_xdata(x_values)
#     # figure.set_xdata(y_values)
#     # figure.set_xlim(0, x_values[-1])
#     # figure.canvas.draw()
#     # figure.canvas.flush_events()
#     plt.scatter(x_values[-1], y_values[-1])
#     plt.pause(0.05)


while True:

    _, frame = cap.read()
    frame = rescale_frame(frame, percent = 30)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    current_time = time.time()
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        EAR_left = (compute(landmarks[37], landmarks[41]) + compute(landmarks[38], landmarks[40]))/(2*compute(landmarks[36],landmarks[39]))
        EAR_right = (compute(landmarks[43],landmarks[47]) + compute(landmarks[44],landmarks[46]))/(2*compute(landmarks[42],landmarks[45]))
        # print(EAR_left, EAR_right)
        #EAR_left = (compute(landmarks[38], landmarks[42]) + compute(landmarks[39], landmarks[41]))/(2*compute(landmarks[37],landmarks[40]))
        #EAR_right = (compute(landmarks[44],landmarks[48]) + compute(landmarks[45],landmarks[47]))/(2*compute(landmarks[43],landmarks[46]))
        EAR = (EAR_left + EAR_right) / 2

        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            cv2.putText(face_frame, str(n), (x,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.putText(face_frame, str(FPS), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255),2)
        # cv2.imshow("Result of detector", face_frame)
    FPS = 1/(current_time-previous_time)
    print(FPS)
    # cv2.imshow("Frame", frame)
    vis.set_y(EAR)
    vis.update()
    previous_time = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

