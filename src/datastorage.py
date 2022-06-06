from itertools import zip_longest
from re import S
import numpy as np
import csv
from datetime import datetime
import time
from json import dumps
from src.Settings import ABS_PATH



class Datastorage():
    def __init__(self):
        data_types = ['Blink', 'Blink_n', 'Perclos', 'Entropy', 'Fatigue', 'Fatigue_Message']
        self.data_dict = self.create_data_dict(data_types)
        
    def set_data(self, type, x, y):
        self.data_dict[type]['x'] = x
        self.data_dict[type]['y'] = y
    
    def safe_data(self, thresholds):
        # Creating file name
        date_time = datetime.fromtimestamp(time.time())
        str_date_time = date_time.strftime("%d_%m_%Y_%H_%M_%S")

        file_name = ABS_PATH + r"/data/data_output_{}.csv".format(str_date_time)
        _fieldnames = self.create_field_names()

        columns_data = self.get_columns()
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(_fieldnames)
            
            column_list = []
            for column in columns_data:
                column_list.append(list(column))
            writer.writerows(column_list)
        
        file_name = ABS_PATH + r"/data/tresholds_{}.txt".format(str_date_time)
        with open(file_name, 'w') as f:
            f.write(dumps(thresholds))


    def create_data_dict(self, types):
        data_dict = {}
        for type in types:
            data_dict[type] = {
                'x': np.array([1,1]),
                'y': np.array([1,1])
            }
        return data_dict
    
    def create_field_names(self):
        fieldnames = []
        for key in self.data_dict.keys():
            for column in ['x','y']:
                fieldnames.append("{}_{}".format(key, column))
        return fieldnames
    
    def get_columns(self):
        columns = []
        for key in self.data_dict.keys():
            for column in ['x','y']:
                columns.append(self.data_dict[key][column])
        return zip_longest(*columns)

        
