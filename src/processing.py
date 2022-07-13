import numpy as np
from cv2 import solvePnP, projectPoints
from scipy.signal import find_peaks
from src.Settings import PROMINENCE, BLINK_WIDTH, FACE_MODEL_MATRIX, CAMERA_MATRIX
import math, cv2

class Processing():
    def __init__(self):
        self.buffer = {'EAR': [],'BLINK': []}
        self.x_values = {'EAR': np.array([]),'BLINK': np.array([])}
        self.y_values = {'EAR': np.array([]),'BLINK': np.array([])}

        self.model_points = np.array(FACE_MODEL_MATRIX)
        self.camera_matrix = np.array(CAMERA_MATRIX, dtype = "double")

        # face direction not towards cam
        self.counter = 0

    def update(self, data):
        orientation_landmarks = np.array(data['orientation'], dtype='float64')
        img = data['img']
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = solvePnP(self.model_points, orientation_landmarks, self.camera_matrix, dist_coeffs, flags=0)
        
        # Projecting 3D point
        (nose_end_point2D, _) = projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, dist_coeffs)
            
        rmat = cv2.Rodrigues(rotation_vector)[0] # rotation matrix
        pmat = np.hstack((rmat, translation_vector)) # projection matrix
        eulers = cv2.decomposeProjectionMatrix(pmat)[-1]
        rol, pitch, yaw = self.get_euler_to_angle(eulers)
        
        p1 = ( int(orientation_landmarks[0][0]), int(orientation_landmarks[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        # for (x, y) in shape:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
        
        # print(ang1, rotation_vector)
        
        cv2.putText(img, "Y: {:.2f}, P: {:.2f}".format(yaw, pitch), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, .6, (128, 255, 255), 3)
        cv2.line(img, p1, p2, (0, 255, 255), 2)

        # If the head is rotated to much on the y-axis dont detect fatigue
        if abs(yaw) > 35 or abs(pitch) < 160: 
            if self.counter >= 100:
                self.counter = 0
                print("[WARNING] The user is not paying attention towards te road")
            self.counter += 1
            return
        self.counter = 0

        # Blinking detection segment     
        (timestamp, landmarks) = data['eye']
        l_landmarks = landmarks['left']
        r_landmarks = landmarks['right']

        EAR_left = (self.distance(l_landmarks[1], l_landmarks[4]) + self.distance(l_landmarks[2], l_landmarks[4]))/(2*self.distance(l_landmarks[0], l_landmarks[3]))
        EAR_right = (self.distance(r_landmarks[1], r_landmarks[4]) + self.distance(r_landmarks[2], r_landmarks[4]))/(2*self.distance(r_landmarks[0], r_landmarks[3]))

        EAR = (EAR_left + EAR_right) / 2
        EAR = 0 if EAR is None else EAR

        self.add_to_buffer('EAR', (timestamp, EAR))

        # Blink segment
        index, properties = self.find_blinks()

        # if list is empty
        if not index.size: return
        #print(min_data, properties)

        temp_y_value = {
            'y': self.y_values['EAR'][index[-1]],
            'left': self.x_values['EAR'][round(properties['left_ips'][-1])],
            'right': self.x_values['EAR'][round(properties['right_ips'][-1])],
            'prominences': properties['prominences'][-1]
        }

        if self.blink_detected(index[-1]):
            self.add_to_buffer('BLINK', (self.x_values['EAR'][index[-1]], temp_y_value))
            print("[DATA] Blink Detected with duration: {}s".format(self.y_values['BLINK'][-1]['right']-self.y_values['BLINK'][-1]['left']))

        if len(self.y_values['EAR'] > 500):
            self.x_values['EAR'] = self.x_values['EAR'][250:]
            self.y_values['EAR'] = self.y_values['EAR'][250:]

    @staticmethod
    def get_euler_to_angle(eulers):
        return (eulers[2][0],eulers[0][0],eulers[1][0])

    def find_blinks(self):
        # if len(self.y_values['EAR']) > 1: print(self.y_values['EAR'][-1:] * -1)
        index, properties = find_peaks(self.y_values['EAR'][-100:] * -1, height=(-0.3, None), prominence=PROMINENCE, width=BLINK_WIDTH)
        
        # Prevents the shifting index from incrementing the blinking frequency
        len_y = len(self.y_values['EAR'])
        if len_y > 100: index += len_y - 100
        # print((index, properties))
        return index, properties

    def blink_detected(self, index):
        if not self.y_values['BLINK'].size: return True
        return self.y_values['BLINK'][-1]['y'] != self.y_values['EAR'][index]


    def get_from_buffer(self, type):
        return self.buffer[type].pop(0)
    
    def add_to_buffer(self, type, element):
        self.x_values[type] = np.append(self.x_values[type], element[0])
        self.y_values[type] = np.append(self.y_values[type], element[1])
        self.buffer[type].append(element)

    def buffer_availble(self, type): 
        return self.buffer[type]
    
    @staticmethod
    def distance(l1, l2):
        return np.linalg.norm(l1 - l2)