import numpy as np
import threading

class Classifier(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.data = {
            'EAR': np.array([])
        }
        pass

    # TODO: Implement ApEn algorithm
    def set_data(self, type, data):
        self.data[type] = data
