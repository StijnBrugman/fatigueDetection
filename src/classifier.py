from cv2 import threshold
import numpy as np
import time, csv
import threading
from datetime import datetime
from src.Settings import ABS_PATH, INIT_TIME, PERCLOS_TIME_INTERVAL, FATIGUE_LEVELS, TRESHOLDS, CLASS_TIMES, CLASS_TRESHOLDS, CLASS_WEIGHT

class Classifier():

    def __init__(self, mode = 'INIT'):
        # threading.Thread.__init__(self)
        self.running = True
        
        self.data = {'EAR': {'x': np.array([]),'y': np.array([]),}, 'BLINK':{ 'x': np.array([]),'y': np.array([]),}}
        self.entropy = {'INIT': np.array([0]), 'RUN': np.array([0])}
        self.n_blink = {'INIT': np.array([0]), 'RUN': np.array([0])}
        self.perclos = {'INIT': np.array([0]), 'RUN': np.array([0])}

        # TODO: This should be a proper strucutre and not this
        self.entropy_time = np.array([0])
        self.perclos_time = np.array([0])
        self.blink_n_time = np.array([0])

        self.timer = time.time()
        self.start_time = time.time()
        self.print_timer = 0

        self.TRESHOLDS =                TRESHOLDS
        self.classifying_times =        CLASS_TIMES
        self.classifying_tresholds =    CLASS_TRESHOLDS
        self.fatigue_message =          FATIGUE_LEVELS
        self.class_weights =            CLASS_WEIGHT

        self.fatigue_dict = {key :  0 for key in self.classifying_times}
        self.fatigue_values = np.array([0])
        self.fatigue_time =  np.array([0])

        self.current_message = self.fatigue_message[0]
        self.previous_message = None

        self.old_perclos = 0
        self.old_n_blink = 0

        self.fatigue_messages = {'x': [], 'y': []}


        self.mode = mode

        self.old_fatigue_level = None
        self.truth_table = []

        self.index = None

        self.key = 'INIT'
        self.init_phase_done = False
        
        self.timers = [0, 0, 0]

        date_time = datetime.fromtimestamp(time.time())
        str_date_time = date_time.strftime("%d_%m_%Y_%H_%M_%S")
        self.file_name = ABS_PATH +  r"/data/data_output_EAR_{}.csv".format(str_date_time)
        
    
    def _run(self):
        # INIT phase
        if not self.initialzed():  self.key = 'INIT'

        

        if not self.init_phase_done and self.initialzed(): 
            print("[INFO] INIT-Phase done. Treshold paramters are: {}".format(self.TRESHOLDS))
            self.init_phase_done = True
            self.key = 'RUN'
        
        
        # Running phase
        
        start_time = time.time()
        (PERCLOS, n_blink) = self.calc_PERCLOS(time_interval = PERCLOS_TIME_INTERVAL)
        self.timers[0] = time.time() - start_time

        start_time = time.time()
        self.update_parameters(PERCLOS, n_blink, self.key)
        self.timers[1] = time.time() - start_time


        # Print segment
        if time.time() - self.print_timer > 5:
            # print(self.timers)
            # print(np.average(self.fatigue_values[-20:]), self.fatigue_level_index)
            self.print_timer = time.time()
            # print(self.truth_table, self.fatigue_dict, self.index)
    
    
    def run(self):
        print("[INFO] Classifier thread Opened")


        key = 'RUN'
        init_phase_done = False
        

        while self.running:
            
            
            # INIT phase
            if not self.initialzed():  key = 'INIT'

            

            if not init_phase_done and self.initialzed(): 
                print("[INFO] INIT-Phase done. Treshold paramters are: {}".format(self.TRESHOLDS))
                init_phase_done = True
                key = 'RUN'
            
            
            # Running phase
            (PERCLOS, n_blink) = self.calc_PERCLOS(time_interval = PERCLOS_TIME_INTERVAL)
            # self.update_parameters(PERCLOS, n_blink, key)

            # Print segment
            if time.time() - self.print_timer > 1:
                # print(np.average(self.fatigue_values[-20:]), self.fatigue_level_index)
                self.print_timer = time.time()
                # print(self.truth_table, self.fatigue_dict, self.index)
        
        print("[INFO] Classifier Thread Closed")

    def update_parameters(self, PERCLOS, n_blink, key):

        # Only run 2 times a second, and when enough data it present
        if time.time() - self.timer > .5 and len(self.data['EAR']['y']) > 50:
                self.timer = time.time()
                modified_list = np.round(self.data['EAR']['y'][-50:] * 100, decimals= 2)
                entropy = self.appr_entropy(modified_list, 2, 3)
                self.entropy[key] = np.append(self.entropy[key], entropy)
                
                if key == 'RUN': self.entropy_time =np.append(self.entropy_time, time.time() - self.start_time)

        self.safe_EAR()


        timestamp = time.time() - self.start_time

        if self.old_n_blink != n_blink:
            self.old_n_blink = n_blink
            self.n_blink[key] = np.append(self.n_blink[key], n_blink)
            if key == 'RUN': self.blink_n_time = np.append(self.blink_n_time, timestamp)

        # TODO: Not a good quick fix
        if self.old_perclos != PERCLOS:
            self.old_perclos = PERCLOS
            self.perclos[key] = np.append(self.perclos[key], PERCLOS)
            if key == 'RUN': self.perclos_time = np.append(self.perclos_time, timestamp)
        # Updating the INIT_TRESHOLDS 
        time_stamp = time.time() - self.start_time
        if time_stamp == 0.0: time_stamp = 0.01
        

        if key == 'INIT':
            entropy      = np.average(self.entropy['INIT'])
            blink        = 30 * len(self.data['BLINK']['x']) / time_stamp
            time_blinked = self.get_time_blinked(self.data['BLINK']['y']) / time_stamp
            
            self.TRESHOLDS = {
                'entropy': entropy, 
                'blink'  : blink,
                'perclos': time_blinked
            }

        else:

            fatigue_level = self.fuzzy_based_classification()

            if self.old_fatigue_level != fatigue_level: 
                self.fatigue_values = np.append(self.fatigue_values, fatigue_level)
                self.fatigue_time = np.append(self.fatigue_time, time_stamp)
                self.old_fatigue_level = fatigue_level

            if len(self.fatigue_values) < 1: return
            
            for time_index in self.classifying_times:
                time_treshold = time.time() - self.start_time - time_index

                # print(self.fatigue_time, time_treshold)

                try:
                    list_index = next(x for x, val in enumerate(self.fatigue_time) if val > time_treshold)
                except StopIteration:
                    break
                
                # print(time_treshold, list_index)
                fatigue_list = self.fatigue_values[list_index:]

                average_fatigue = np.average(fatigue_list)
                self.fatigue_dict[time_index] = average_fatigue

            self.index = self.classify_case_statement()
            self.current_message = self.fatigue_message[self.index]

            if self.current_message !=  self.previous_message:
                self.fatigue_messages['x'].append(time_stamp)
                self.fatigue_messages['y'].append(self.current_message)
                print("[INFO] Fatigue level: {} ".format(self.current_message))  
            
            self.previous_message = self.current_message
    

    def safe_EAR(self):
        if len(self.data['EAR']['y']) > 500:
            with open(self.file_name, 'a') as f:
                writer = csv.writer(f)
                for (x, y) in zip(self.data['EAR']['x'][:450], self.data['EAR']['y'][:450]):
                    writer.writerow([x, y])
            self.data['EAR']['x'] = self.data['EAR']['x'][450:]
            self.data['EAR']['y'] = self.data['EAR']['y'][450:]

    def get_time_blinked(self, dataset):
        time_blinked = 0
        for y  in dataset:
            width = y['right'] - y['left']
            time_blinked += width
        return time_blinked

    def create_true_table(self):
        table = []
        for i, key in enumerate(self.fatigue_dict):
            row = []
            for j, threshold in enumerate(self.classifying_tresholds):
                if self.fatigue_dict[key] > threshold:
                    row.append(5 - i - j)
                else:
                    row.append(-1)
            table.append(row)
        return np.array(table)
            
    def classify_case_statement(self):
        table = self.create_true_table()

        # Convert Matrix to Priority Array
        priority_array = np.hstack([table[::-1].diagonal(offset=x) \
                for x in np.arange(-len(table)+1,len(table[0]))])
        
        # Storage element
        self.truth_table = priority_array, table

        # Find first non-zero element for message
        for value in priority_array:
            if value != -1: return value
        return 0

    def stop(self):
        self.running = False
    
    def get_blink(self):
        return [data_points['y'] for data_points in self.data['BLINK']['y']]


    
    def set_data(self, type, data):
        x, y = data
        self.data[type]['x'] = np.append(self.data[type]['x'], x)
        self.data[type]['y'] = np.append(self.data[type]['y'], y)

    def fuzzy_based_classification(self):
        # Blinking
        percentage = self.n_blink['RUN'][-1] / self.TRESHOLDS['blink']
        value_1 = self.mapping(percentage, 1, 2, 0, self.class_weights['blink'])
        
        # Entropy
        entropy = self.entropy['RUN'][-1] / self.TRESHOLDS['entropy']
        value_2 = self.mapping(entropy, 1, 2, 0, self.class_weights['entropy'])
        
        # PERCLOS
        perclos = self.perclos['RUN'][-1] / self.TRESHOLDS['perclos']
        value_3 = self.mapping(perclos, 1, 2, 0, self.class_weights['perclos'])
        return value_1 + value_2 + value_3

    def initialzed(self):
        return ((time.time() - self.start_time) > INIT_TIME) or self.mode
    
    def calc_PERCLOS(self, time_interval = 30):
        min_time = time.time() - self.start_time - time_interval
        x = self.data['BLINK']['x']
        x_values = x[x > min_time]

        # if no blinks detected
        len_x = len(x_values)
        if not len_x: return 0, 0

        time_eye_closed = self.get_time_blinked(self.data['BLINK']['y'][-(len_x):])
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