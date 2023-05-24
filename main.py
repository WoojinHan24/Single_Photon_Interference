import single_photon_interference as spi
import numpy as np
import pandas as pd
import pickle
import re
import warnings


warnings.filterwarnings(action='ignore')


experiments = ['laser_experiments', 'PMT_upper_boundary','PMT_lower_boundary','Thershold_validity','Sensor_slit_position','single_photon_interference']

try:
    with open("datum.pkl","rb") as f:
        datum = pickle.load(f)


except FileNotFoundError:
    raw_data_file_name = "./spi_raw_data.xlsx"
    df = pd.read_excel(raw_data_file_name, sheet_name= None)
    sheets = df.keys()

    datum={experiment : [] for experiment in experiments}

    for sheet in sheets:
        data_list = [] 
        data_set_conditions={}
        dataframe=df[sheet]
        print(sheet)

        if sheet[0] <= '9' and sheet[0] >= '0':
            experiment = "laser_experiments"
            
            column_dic = dataframe.columns
            conditions_position=dataframe.columns.get_loc('conditions')
            for index in range(dataframe.shape[0]):
                data_results = []
                for column in column_dic:
                    measurement = dataframe.loc[index, column]

                    if column == 'position(cm)':
                        data_parameter = np.float64(measurement)
                    if column == 'Voltage(mV)':
                        data_results.append(np.float64(measurement))
                    if column == 'conditions' and pd.isnull(measurement) == False:
                        data_set_conditions[measurement] = dataframe.iloc[index,conditions_position+1]

                if data_results != []:
                    data = spi.Data(data_parameter,data_results)
                    data_list.append(data)

            
            data_set = spi.Data_set(data_set_conditions,['position(cm)'],['Voltage(mV)'],data_list)
            datum[experiment].append(data_set)

        elif 'PMT-Upper' == sheet[0:9]:
            experiment = 'PMT_upper_boundary'

            column_dic = dataframe.columns
            conditions_position=dataframe.columns.get_loc('conditions')

            for column in column_dic:

                data_results = []
                for index in range(dataframe.shape[0]):

                    measurement = dataframe.loc[index, column]

                    try:
                        data_parameter = np.float64(column)
                        data_results.append(np.float64(measurement))

                    except ValueError:
                        if column == 'conditions' and pd.isnull(measurement) == False:
                            data_set_conditions[measurement] = dataframe.iloc[index,conditions_position+1]
                        
                if data_results != []:
                    data = spi.Data(data_parameter,data_results)
                    data_list.append(data)
            
            data_set = spi.Data_set(data_set_conditions,['High_Voltage(V)'],[f'Count{trial}' for trial in range(1,21)],data_list)
            datum[experiment].append(data_set)
        
        elif 'PMT-Lower' == sheet[0:9] or 'PMT_Lower' == sheet[0:9]:
            experiment = 'PMT_lower_boundary'
            column_dic = dataframe.columns
            conditions_position=dataframe.columns.get_loc('conditions')

            for column in column_dic:

                data_results = []
                for index in range(dataframe.shape[0]):

                    measurement = dataframe.loc[index, column]

                    try:
                        data_parameter = np.float64(column)
                        data_results.append(np.float64(measurement))

                    except ValueError:
                        if column == 'conditions' and pd.isnull(measurement) == False:
                            data_set_conditions[measurement] = dataframe.iloc[index,conditions_position+1]
                        
                if data_results != []:
                    data = spi.Data(data_parameter,data_results)
                    data_list.append(data)
            
            data_set = spi.Data_set(data_set_conditions,['High_Voltage(V)'],[f'Count{trial}' for trial in range(1,21)],data_list)
            datum[experiment].append(data_set)
        
        elif 'Threshold' == sheet:
            experiment = 'Thershold_validity'

            column_dic = dataframe.columns
            data_parameter = [674,0.8]
            for index in range(dataframe.shape[0]):
                data_results = []
                for column in column_dic:
                    measurement = dataframe.loc[index, column]

                    if column == 'Count(PCIT)' or column == 'Count(Oscilloscope-18.4mV)' or column == 'Count(Oscilloscope-17.6mV)' or column == 'Count(Oscilloscope-19.2mV)':
                        data_results.append(np.float64(measurement))

                    
                        
                if data_results != []:
                    data = spi.Data(data_parameter,data_results)
                    data_list.append(data)
            
            data_set = spi.Data_set(data_set_conditions,['High Voltage [V]', 'Threshold(PCIT)'],['Count(Oscilloscope-18.4mV)','Count(Oscilloscope-17.6mV)','Count(Oscilloscope-19.2mV)'],data_list)
            datum[experiment].append(data_set)

        elif 'Sensor Slit Position' == sheet[0:20] or 'Double Slit' ==sheet[0:11]:
            experiment = "Sensor_slit_position"

            if 'Double Slit' == sheet[0:11]:
                experiment = 'single_photon_interference'
            
            column_dic = dataframe.columns
            conditions_position=dataframe.columns.get_loc('conditions')
            for index in range(dataframe.shape[0]):
                data_results = []
                for column in column_dic:
                    measurement = dataframe.loc[index, column]

                    if column == 'Sensor Slit Position Position ':
                        data_parameter = np.float64(measurement)
                    if column[0:5] == 'Count':
                        data_results.append(np.float64(measurement))
                    if column == 'conditions' and pd.isnull(measurement) == False:
                        data_set_conditions[measurement] = dataframe.iloc[index,conditions_position+1]

                if data_results != []:
                    data = spi.Data(data_parameter,data_results)
                    data_list.append(data)

            data_set = spi.Data_set(data_set_conditions,['sensor_slit_position (cm)'],[f'Count{trial}'for trial in range(1,8)],data_list)
            datum[experiment].append(data_set)


    with open("./datum.pkl", "wb") as f:
        pickle.dump(datum,f)
            

       
for experiment in experiments:
    for data_set in datum[experiment]:
        print(experiment)
        data_set.print_data()