from itertools import zip_longest
from re import S
import numpy as np
import csv

class Datastorage():
    def __init__(self):
        data_types = ['EAR', 'Blink', 'Entropy', 'Fatigue']
        self.data_dict = self.create_data_dict(data_types)
        
    def set_data(self, type, x, y):
        self.data_dict[type]['x'] = x
        self.data_dict[type]['y'] = y
    
    def safe_data(self):
        file_name = "data_output_{}".format("test")
        _fieldnames = self.create_field_names()

        columns_data = self.get_columns()
        with open(file_name, 'w') as file_name:
            writer = csv.writer(file_name)
            writer.writerow(_fieldnames)
            
            column_list = []
            for column in columns_data:
                column_list.append(list(column))
            print(column_list)
            writer.writerows(column_list)

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
        
