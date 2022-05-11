import numpy as np
from scipy.signal import find_peaks
from Settings import PROMINENCE


class Processing():
    def __init__(self):
        self.buffer = {
            'EAR': [],
            'BLINK': []
        }

        self.x_values = {
            'EAR': np.array([]),
            'BLINK': np.array([])
        }
        self.y_values = {
            'EAR': np.array([]),
            'BLINK': np.array([])
        }

    def update(self, data):
        (timestamp, landmarks) = data
        print(timestamp)
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


        # print("x_values", len(self.x_values['EAR']))
       
        # print(self.x_values['EAR'][index[-1]])
        if self.blink_detected(index[-1]):
            self.add_to_buffer('BLINK', (self.x_values['EAR'][index[-1]], temp_y_value))
            print("[DATA] Blink Detected with paramters: {}".format(self.y_values['BLINK'][-1]))
            print(self.x_values['BLINK'])

    def find_blinks(self):
        return find_peaks(self.y_values['EAR'][-1000:] * -1, height=(None, 0.3), prominence=PROMINENCE, width=0.2)

    

    def blink_detected(self, index):
        if not self.y_values['BLINK'].size: return True
        # print("Comparison", self.y_values['BLINK'], self.y_values['EAR'][index])
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
    
#     EAR_left = (compute(landmarks[37], landmarks[41]) + compute(landmarks[38], landmarks[40]))/(2*compute(landmarks[36],landmarks[39]))
#     EAR_right = (compute(landmarkqqqs[43],landmarks[47]) + compute(landmarks[44],landmarks[46]))/(2*compute(landmarks[42],landmarks[45]))
#     # print(EAR_left, EAR_right)
#     #EAR_left = (compute(landmarks[38], landmarks[42]) + compute(landmarks[39], landmarks[41]))/(2*compute(landmarks[37],landmarks[40]))
#     #EAR_right = (compute(landmarks[44],landmarks[48]) + compute(landmarks[45],landmarks[47]))/(2*compute(landmarks[43],landmarks[46]))
#     EAR = (EAR_left + EAR_right) / 2