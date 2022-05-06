import numpy as np

class Processing():
    def __init__(self):
        self.buffer = []

    def update(self, data):
        (timestamp, landmarks) = data
        l_landmarks = landmarks['left']
        r_landmarks = landmarks['right']

        EAR_left = (self.distance(l_landmarks[1], l_landmarks[4]) + self.distance(l_landmarks[2], l_landmarks[4]))/(2*self.distance(l_landmarks[0], l_landmarks[3]))
        EAR_right = (self.distance(r_landmarks[1], r_landmarks[4]) + self.distance(r_landmarks[2], r_landmarks[4]))/(2*self.distance(r_landmarks[0], r_landmarks[3]))

        EAR = (EAR_left + EAR_right) / 2
        EAR = 0 if EAR is None else EAR

        self.add_to_buffer((timestamp, EAR))
    


    def get_from_buffer(self):
        return self.buffer.pop(0)
    
    def add_to_buffer(self, element):
        self.buffer.append(element)

    def buffer_availble(self):
        return self.buffer
    
    @staticmethod
    def distance(l1, l2):
        return np.linalg.norm(l1 - l2)
    
#     EAR_left = (compute(landmarks[37], landmarks[41]) + compute(landmarks[38], landmarks[40]))/(2*compute(landmarks[36],landmarks[39]))
#     EAR_right = (compute(landmarks[43],landmarks[47]) + compute(landmarks[44],landmarks[46]))/(2*compute(landmarks[42],landmarks[45]))
#     # print(EAR_left, EAR_right)
#     #EAR_left = (compute(landmarks[38], landmarks[42]) + compute(landmarks[39], landmarks[41]))/(2*compute(landmarks[37],landmarks[40]))
#     #EAR_right = (compute(landmarks[44],landmarks[48]) + compute(landmarks[45],landmarks[47]))/(2*compute(landmarks[43],landmarks[46]))
#     EAR = (EAR_left + EAR_right) / 2