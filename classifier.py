from cv2 import threshold
import numpy as np
import time
import threading
from Settings import INIT_TIME, PERCLOS_TIME_INTERVAL, FATIGUE_LEVELS, TRESHOLDS

class Classifier(threading.Thread):

    def __init__(self, mode = 'INIT'):
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
            'INIT': np.array([0]),
            'RUN': np.array([0])
        }
        self.n_blink = {
            'INIT': np.array([0]),
            'RUN': np.array([0])
        }
        self.perclos = {
            'INIT': np.array([0]),
            'RUN': np.array([0])
        }

        self.TRESHOLDS = TRESHOLDS

        self.init_phase_done = False

        self.classifying_times = [60, 45, 25, 5]
        self.classifying_tresholds = [70, 55, 35]
        self.fatigue_message = FATIGUE_LEVELS

        self.fatigue_values = np.array([])
        self.fatigue_time =  np.array([])

        self.current_message = self.fatigue_message[0]
        self.previous_message = None

        (self.old_PERCLOS, self.old_n_blink) = (None, None)

        self.mode = mode

        self.old_fatigue_level = None

        

    def run(self):
        print("[INFO] Classifier thread Opened")
        
        while self.running:
            key = 'RUN'

            # INIT phase
            if not self.initialzed():  key = 'INIT'

            if not self.init_phase_done and self.initialzed(): 
                print("[INFO] INIT-Phase done. Treshold paramters are: {}".format(self.TRESHOLDS))
                self.init_phase_done = True

            
            # Running phase
            (PERCLOS, n_blink) = self.calc_PERCLOS(time_interval = PERCLOS_TIME_INTERVAL)
            
            
            self.update_parameters(PERCLOS, n_blink, key)

            # Printing every ... second
            if time.time() - self.print_timer > .5:
                # print(np.average(self.fatigue_values[-20:]), self.fatigue_level_index)
                print(self.current_message)
                self.print_timer = time.time()
                #  print(self.TRESHOLDS)
        print("[INFO] Classifier Thread Closed")

    def update_parameters(self, PERCLOS, n_blink, key):
        # print(len(self.data['EAR']['y']))
        if time.time() - self.timer > .5 and len(self.data['EAR']['y']) > 200:
                self.timer = time.time()
                modified_list = np.round(self.data['EAR']['y'][-200:] * 100, decimals= 2)
                entropy = self.appr_entropy(modified_list, 2, 3)
                self.entropy[key] = np.append(self.entropy[key], entropy)
            
        self.n_blink[key] = np.append(self.n_blink[key], n_blink)
        self.perclos[key] = np.append(self.perclos[key], PERCLOS)

        # Updating the INIT_TRESHOLDS 
        if key == 'INIT':
            time_spent = (time.time() - self.start_time)
            time_constant = time_spent / 30
            blinking_n = len(self.data['BLINK']['x'])

            time_blinked = 0
            for y  in self.data['BLINK']['y']:
                width = y['right'] - y['left']
                time_blinked += width
            
            self.TRESHOLDS = {
                'entropy': np.average(self.entropy['INIT']), 
                'blink': blinking_n / time_constant,
                'perclos': time_blinked / time_spent   
            }
        else:

            # if (PERCLOS, n_blink) ==  (self.old_PERCLOS, self.old_n_blink): return
            # (self.old_PERCLOS, self.old_n_blink) =  (PERCLOS, n_blink)

            # print(self.n_blink[key][-1], self.perclos[key][-1], self.entropy[key][-1])

            # Only adding a fatigue_level if it has changed
            fatigue_level = self.fuzzy_based_classification()
            time_stamp = time.time() - self.start_time
            if self.old_fatigue_level != fatigue_level: 
                self.fatigue_values = np.append(self.fatigue_values, fatigue_level)
                self.fatigue_time = np.append(self.fatigue_time, time_stamp)
            self.old_fatigue_level = fatigue_level

            """
            self.classifying_times = [5, 10, 20, 40]
            self.classifying_tresholds = [50, 60, 70]
            """

            # TODO: Get the correct fatigue values over time
            # print(self.fatigue_values[-1])
            message_set = False

            fatigue_dict = {}
            for value in self.classifying_times:
                fatigue_dict[value] = 0

            

            if len(self.fatigue_values) < 1: return
            self.fatigue_level_index = 0
            for i, time_index in enumerate(self.classifying_times):
                time_treshold = time.time() - self.start_time - time_index

                # print(self.fatigue_time, time_treshold)

                try:
                    list_index = next(x for x, val in enumerate(self.fatigue_time) if val > time_treshold)
                except StopIteration:
                    break
                
                # print(time_treshold, list_index)
                fatigue_list = self.fatigue_values[list_index:]
                
                # print(fatigue_list)
                average_fatigue = np.average(fatigue_list)
                fatigue_dict[time_index] = average_fatigue

                # for j, threshold in enumerate(self.classifying_tresholds):
                #     if average_fatigue > threshold:
                #         message_set = True
                        
                #         self.fatigue_level_index = 6 - i -j
                #         print(average_fatigue, self.fatigue_level_index)
                #         self.current_message = self.fatigue_message[self.fatigue_level_index]
                #         break
                # else:
                #     continue
                # break
            # print(fatigue_dict)

            index = self.classify_case_statement(fatigue_dict)
            self.current_message = self.fatigue_message[index]
            
            # if not message_set: self.current_message = self.fatigue_message[0]

            if self.current_message !=  self.previous_message:
                print("[INFO] Fatigue level: {} ".format(self.current_message))  
            
            self.previous_message = self.current_message
        
    def create_true_table(self, fatigue_dict):
        table = []
        for i, key in enumerate(fatigue_dict):
            row = []
            for j, threshold in enumerate(self.classifying_tresholds):
                if fatigue_dict[key] > threshold:
                    row.append(6 - i - j)
                else:
                    row.append(-1)
            table.append(row)
        
        
        # table = np.delete(table, 0)
        return table
            
    def classify_case_statement(self, fatigue_dict):
        index = 0

        table = self.create_true_table(fatigue_dict)
        table = np.array(table)

        # table = np.arange(np.prod(table)).reshape(table)
        out = np.hstack([table[::-1].diagonal(offset=x) \
                for x in np.arange(-len(table)+1,len(table[0]))])
        # print(table,out)

        for value in out:
            
            if value != -1:
                # print(value, out)
                return value
        
        return index

    def extract_list(self, l):
        print(l)
        return [item[0] for item in l]

    def stop(self):
        self.running = False
    
    # TODO: Implement ApEn algorithm
    def set_data(self, type, data):
        x, y = data
        self.data[type]['x'] = np.append(self.data[type]['x'], x)
        self.data[type]['y'] = np.append(self.data[type]['y'], y)
        # print(self.data)

    def fuzzy_based_classification(self):
        # Blinking
        
        
        percentage = self.n_blink['RUN'][-1] / self.TRESHOLDS['blink']
        value_1 = self.mapping(percentage, 0.5, 3, 0, 20)
        # print(self.n_blink['RUN'][-1], self.TRESHOLDS['blink'], value_1)

        # Entropy
        entropy = self.entropy['RUN'][-1] / self.TRESHOLDS['entropy']
        value_2 = self.mapping(entropy, 0.5, 2, 0, 40)
        # print(self.entropy['RUN'][-1], self.TRESHOLDS['entropy'], value_2)

        # PERCLOS
        perclos = self.perclos['RUN'][-1] / self.TRESHOLDS['perclos']
        value_3 = self.mapping(perclos, 0.5, 3, 0, 40)

        # print(value_1, value_2, value_3)

        # value between 0 - 100 indicating how fatigued a person is
        # print(value_1, value_2, value_3)
        return value_1 + value_2 + value_3


    def initialzed(self):
        # if self.mode: self.TRESHOLDS = TRESHOLDS
        return ((time.time() - self.start_time) > INIT_TIME) or self.mode
    
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
    

    @staticmethod
    def mapping(number, r1_min, r1_max, r2_min, r2_max):
        if number > r1_max: return r2_max
        if number < r1_min: return r2_min
        r1_width = r1_max - r1_min
        r2_width = r2_max - r2_min
        factor = (number-r1_min) / r1_width
        return (r2_width * factor) + r2_min