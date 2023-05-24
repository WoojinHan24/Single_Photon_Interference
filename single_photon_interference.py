import numpy as np
import matplotlib.pyplot as plt

class Data:

    def __init__(self,parameters,results):
        self.parameters = parameters
        self.results = results

    
    def print_data(self):
        print("parameters : ", self.parameters)
        print("results : ", self.results)
        

class Data_set:
    align_condition :dict
    parameters_name :list
    results_name : list
    data_list : list
    
    def __init__(self, align_condition,parameters_name, results_name,data_list = []):
        self.align_condition =align_condition
        self.parameters_name = parameters_name
        self.results_name = results_name
        self.data_list = data_list
        
    def data_append(self, data):
        self.data_list.append(data)

    def print_data(self, index_list = []):
        if index_list == []:
            index_list = [0,1,-2,-1]
        
        print("Align Condition", self.align_condition)
        print("In parameters ", self.parameters_name)
        print("Measured results ", self.results_name)
        for index in index_list:
            self.data_list[index].print_data()