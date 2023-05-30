import single_photon_interference as spi
import scipy.integrate as integ
import numpy as np
from scipy.constants import pi
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


experiment = 'laser_experiments'
data_set_list = datum[experiment]
#for data_set in data_set_list:
#    data_set.print_data()
# s_l value is the FWHM/wavelength value. which di's both share

def double_slit_modified_function(
    x,c,I,d1,d2,s_l
):
    if x is list:
        return list(map(lambda a: double_slit_modified_function(a,c,I,d1,d2,s_l),x))
    
    N=1000
    a = np.linspace(-5*s_l,5*s_l,N)
    z=list(map(lambda b: double_slit_fitting_function(x,c,I,d1*(1+b),d2*(1+b),I)*lorentzian(b,s_l),a))
    return sum(z)*10*s_l/N
    

def lorentzian(
    x,fwhm
):
    return 1/pi * (1/2*fwhm)/(x**2 + (1/2 * fwhm)**2)

def single_slit_modified_function(
    x,c,I,d1,s_l
):
    if x is list:
        return list(map(lambda a: single_slit_modified_function(a,c,I,d1,s_l),x))
    N = 1000
    a = np.linspace(-5*s_l,5*s_l,N)
    z = list(map(lambda b: single_slit_fitting_function(x,c,I,d1*(1+b))*lorentzian(b,s_l),a))
    return sum(z) *10*s_l/N

def double_slit_asymmetry_fitting_function(
    x, c, I, d1,d2,I2
):
    if type(x) == list:
        return list(map(lambda a: double_slit_asymmetry_fitting_function(a,c,I,d1,d2,I2),x))
    
    alpha = pi*(x-c)/(d1)
    beta = pi* (x-c)/(d2)


    return (I+I2 + 2*np.sqrt(I*I2)*np.cos(2*beta))/4 *(np.sinc(alpha))**2


def double_slit_fitting_function(
    x, c, I, d1,d2,I2
):
    if type(x) == list:
        return list(map(lambda a: double_slit_fitting_function(a,c,I,d1,d2,I2),x))
        
    alpha = pi*(x-c)/(d1)
    beta = pi* (x-c)/(d2)


    
    return I*(np.cos(beta))**2 *(np.sinc(alpha))**2

def double_slit_rough_fitting_function(
    x, c, I, d1,d2,I2
):
    if type(x) == list:
        return list(map(lambda a: double_slit_rough_fitting_function(a,c,I,d1,d2,I2),x))
    alpha = pi*(x-c)/(d1)
    beta = pi* (x-c)/(d2)

    return I*(np.cos(beta))**2
    
def single_slit_fitting_function(
    x,c, I, d1
):
    alpha = pi * (x-c)/d1

    return I * (np.sinc(alpha))**2

def single_slit_rough_fitting_function(
    x,c,I,d1
):
    alpha = pi*(x-c)/d1

    return I*(1- alpha**2/6)**2
    

def laser_double_slit_param_setting(
    x_list, y_list
):
    c,I = max(zip(x_list,y_list),key = lambda pair:pair[1])
    
    crit_list = []
    for index in range(len(x_list)-2):
        if (y_list[index+1]-y_list[index])*(y_list[index+2]-y_list[index+1]) <0:
            crit_list.append(index+1)

    for index in range(len(crit_list)):
        if x_list[crit_list[index]] == c:
            d2 = x_list[crit_list[index+1]] - x_list[crit_list[index-1]]

    d1 = 0.5
    return [c,I,d1,d2,I]



wavelength = 689.7*1e-9
L=0.33


for align in range(1,7):
    for exp_type in ['double_slit']:
        d = 2.7*1e-6
        a = 5*1e-7
        laser_raw_fig = spi.phys_plot(
            data_set_list,
            lambda x: x.parameters,
            lambda x: x.results[0],
            {'align' : align, 'exp_type' : exp_type},
            x_label = "position [cm]",
            y_label = "voltage [mV]",
            labels = lambda x: f"{x.align_condition['align']}" + x.align_condition['exp_type']+f"_{x.align_condition['trial']}",
            fitting_function= double_slit_asymmetry_fitting_function,
            rough_fitting_functions = [double_slit_rough_fitting_function,double_slit_fitting_function],
            p0_function = laser_double_slit_param_setting,
            truncate = lambda x: True if x<0.7 else False,
            export_param_statics = f"./results/laser({align}_{exp_type})_raw_param_statics.txt"
        )
        
        try:
            laser_raw_fig.savefig(f"./results/laser({align}_{exp_type})_raw_fig.png")
        except AttributeError:
            continue
            

for align in range(1,7):
    for exp_type in ['R_single_slit', 'L_single_slit']:
        laser_raw_fig = spi.phys_plot(
            data_set_list,
            lambda x: x.parameters,
            lambda x: x.results[0],
            {'align' : align, 'exp_type' : exp_type},
            x_label = "position [cm]",
            y_label = "voltage [mV]",
            labels = lambda x: f"{x.align_condition['align']}" + x.align_condition['exp_type']+f"_{x.align_condition['trial']}",
            fitting_function=single_slit_fitting_function,
            rough_fitting_functions = [single_slit_rough_fitting_function],
            p0 = [0.36,505,0.49],
            truncate = lambda x: True if x<0.8 else False,
            export_param_statics = f"./results/laser({align}_{exp_type})_raw_param_statics.txt"
        )
        
        try:
            laser_raw_fig.savefig(f"./results/laser({align}_{exp_type})_raw_fig.png")
        except AttributeError:
            continue


for align in range(1,7):
    for exp_type in ['double_slit']:
        d = 2.7*1e-6
        a = 5*1e-7
        laser_modified_fig = spi.phys_plot(
            data_set_list,
            lambda x: x.parameters,
            lambda x: x.results[0],
            {'align' : align, 'exp_type' : exp_type},
            x_label = "position [cm]",
            y_label = "voltage [mV]",
            labels = lambda x: f"{x.align_condition['align']}" + x.align_condition['exp_type']+f"_{x.align_condition['trial']}",
            fitting_function= double_slit_modified_function,
            rough_fitting_functions = [double_slit_rough_fitting_function,double_slit_fitting_function],
            fitting_param_query = [None,lambda x: [*x[:4],1e-2]],
            p0_function = laser_double_slit_param_setting,
            truncate = lambda x: True if x<0.7 else False,
            export_param_statics = f"./results/laser({align}_{exp_type})_param_statics.txt"
        )
        
        try:
            laser_modified_fig.savefig(f"./results/laser({align}_{exp_type})_modified_fig.png")
        except AttributeError:
            continue
            

for align in range(1,7):
    for exp_type in ['R_single_slit', 'L_single_slit']:

        laser_modified_fig = spi.phys_plot(
            data_set_list,
            lambda x: x.parameters,
            lambda x: x.results[0],
            {'align' : align, 'exp_type' : exp_type},
            x_label = "position [cm]",
            y_label = "voltage [mV]",
            labels = lambda x: f"{x.align_condition['align']}" + x.align_condition['exp_type']+f"_{x.align_condition['trial']}",
            fitting_function= single_slit_modified_function,
            rough_fitting_functions = [single_slit_rough_fitting_function,single_slit_fitting_function],
            fitting_param_query = [None,lambda x: [*x,0.01]],
            p0 = [0.36,505,0.49],
            truncate = lambda x: True if x<0.8 else False,
            export_param_statics = f"./results/laser({align}_{exp_type})_param_statics.txt"
        )
        
        try:
            laser_modified_fig.savefig(f"./results/laser({align}_{exp_type})_modified_fig.png")
        except AttributeError:
            continue