import numpy as np
import time
import threading
from Settings import INIT_TIME, PERCLOS_TIME_INTERVAL

class Classifier(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True

        self.start_time = time.time()

        self.data = {
            'EAR': {
                'x': np.array([]),
                'y': np.array([]),
            },
            'BLINK': { 
                'x': np.array([]),
                'y': np.array([]),
            }
        }

        self.timer = time.time() + 5
        self.print_timer = 0

        self.entropy = {
            'INIT': np.array([]),
            'RUN': np.array([])
        }
        self.n_blink = {
            'INIT': np.array([]),
            'RUN': np.array([])
        }
        self.perclos = {
            'INIT': np.array([]),
            'RUN': np.array([])
        }
        

    def run(self):
        print("[INFO] Classifier thread Opened")
        while self.running:
            key = 'RUN'

            # INIT phase
            if not self.initialzed():  key = 'INIT'
            
            # Running phase
            (PERCLOS, n_blink) = self.calc_PERCLOS(time_interval = PERCLOS_TIME_INTERVAL)
            
            if time.time() - self.timer > .5:
                self.timer = time.time()
                modified_list = np.round(self.data['EAR']['y'][-100:] * 100, decimals= 2)
                entropy = self.appr_entropy(modified_list, 2, 3)
                self.entropy[key] = np.append(self.entropy[key], entropy)
            
            self.n_blink[key] = np.append(self.n_blink[key], n_blink)
            self.perclos[key] = np.append(self.perclos[key], PERCLOS)

            # print(entropy)
            if time.time() - self.print_timer > 2:
                self.print_timer = time.time()

                # time_constant = (time.time() - self.start_time) 

                # print(np.average(self.entropy['RUN']), np.average((self.n_blink['RUN'])), np.average(self.perclos['RUN']))
        print("[INFO] Classifier Thread Closed")

    def stop(self):
        self.running = False
    
    # TODO: Implement ApEn algorithm
    def set_data(self, type, data):
        x, y = data
        self.data[type]['x'] = np.append(self.data[type]['x'], x)
        self.data[type]['y'] = np.append(self.data[type]['y'], y)
        # print(self.data)

    def initialzed(self):
        return (time.time() - self.start_time) > INIT_TIME
    
    def calc_PERCLOS(self, time_interval = 30):
        current_time = time.time() - self.start_time
        min_time = current_time - time_interval
        x = self.data['BLINK']['x']
    
        x_values = x[x > min_time]
        # print(x_values)
        # If no blinking has happend PERCLOS = 0.0
        if not len(x_values): return 0, 0

        len_x = len(x_values)
        time_eye_closed = 0

        for y in self.data['BLINK']['y'][-(len_x-1):]:
            time_eye_closed += y['right'] - y['left']
        
        # print(time_eye_closed)

        PERCLOS = time_eye_closed/time_interval

        return PERCLOS, len_x

    def appr_entropy(self, sequence, m, r):

        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[sequence[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [
                len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                for x_i in x
            ]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))

        N = len(sequence)

        return _phi(m) - _phi(m + 1)




    # init phase
    # ApEN calc
    # PERCLOS calc
    # Blink calc