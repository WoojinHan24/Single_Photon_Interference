import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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
            
            
class Plot_element:
    def __init__(self, x , y, fmt, x_err=None,y_err=None,label=None,param=None,param_cov=None, x_continuous=None, R_square =None):
        self.x = x
        self.y = y
        self.fmt = fmt
        self.xerr = x_err
        self.yerr = y_err
        self.label = label
        self.param = param
        self.param_cov = param_cov
        self.x_continuous = x_continuous
        self.R_square = R_square
        
    def get_coef(self):
        return (self.x,self.y,self.fmt,self.xerr, self.yerr,self.label, self.param, self.x_continuous)
    
    def get_param_statics(self):
        table = self.label + "& "
        
        for p,perr in zip(self.param, np.diag(self.param_cov)):
            table = table + f"${p} \pm {perr}$" + "& "
        
        table = table + f"{self.R_square}" + "\\\\ \hline \n"
        
        return table



def phys_plot(
    data_set_list,
    x_function,
    y_function,
    selection,
    x_label,
    y_label,
    fmts = ['k.','b.','y.','c.'],
    error_x = None,
    error_y = None,
    labels = None,
    fitting_function = None,
    rough_fitting_functions = None,
    fitting_param_query = [None,None,None,None],
    p0 = None,
    p0_function = None,
    truncate = None,
    export_param_statics =None
):
    #data_set_list : list of spi.Data_set() in same experiment
    #x_function : the function outputs x_variable in the plot, in function of spi.Data() type
    #y_function : the function outputs y_variable in the plot, in function of spi.Data() type
    #selection : the filter dictionary of data_set_list, which can filter out the expected measurements in the data_set_list
    #           therefore, the function plots the spi.Data_set()s which matches the selection dictioray in arbitrary fmt type.
    #fmts : format of the plots in the numerates of the inputs
    #error_x : error value of x in function of spi.Data() element, it plots error bar in x direction
    #error_y : error value of y in function of spi.Data() element, it plots error bar in y direction
    #labels : This option puts legend with respect to the function of spi.Data_set() expected to contribute with spi.Data_set.align_condition
    #fitting_function and p0 : fits the curves in function of fitting function, in initial parameters of p0.
    #rough_fitting_function : function fitting queries not to fit in local minima
    #fitting_param_query : the query of parameter fixing while the roughly fitting is proceeding
    #truncate : limit fitting regi
    #export param statics : export parameters in form of LaTex table, leave blank in each parameters
    
    
    fig = plt.figure(figsize = (8,4))
    ax = fig.add_subplot(1,1,1)


    plot_list = []    
    for data_set in data_set_list:
        if dictionary_boolean(data_set.align_condition,selection) == True:
            x=[x_function(data) for data in data_set.data_list]
            y=[y_function(data) for data in data_set.data_list]
            x_err=None
            y_err=None
            label=None
            param = p0
            param_cov = None
            x_continuous = None
            R_square = None
            
            if error_x is not None:
                x_err = [error_x(data) for data in data_set.data_list]
            if error_y is not None:
                y_err = [error_y(data) for data in data_set.data_list]
            
            if labels is not None:
                label = labels(data_set)
            
            if fitting_function is not None:
                
                x_fit = np.array([x_val for x_val in x if truncate == None or truncate(x_val)==True])
                y_fit = np.array([y_val for x_val,y_val in zip(x,y) if  truncate == None or truncate(x_val)==True])

                if p0_function is not None:
                    param = p0_function(x_fit, y_fit)
                
                if rough_fitting_functions is not None:
                    for rough_fitting_function, param_function in zip(rough_fitting_functions, fitting_param_query):
                        param,_ = get_regression_result(x_fit, y_fit, rough_fitting_function,param,1000)
                        if param_function is not None:
                            param = param_function(param)

                
                param,param_cov = get_regression_result(x_fit,y_fit,fitting_function,param)
                x_continuous = np.linspace(min(x),max(x),500)
                residuals = y_fit - fitting_function(x_fit,*param)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_fit-np.mean(y_fit))**2)
                R_square= 1-ss_res/ss_tot
                
            
            
            plot_list.append(Plot_element(x,y,fmts[len(plot_list)],x_err=x_err,y_err=y_err,label=label,param=param,param_cov=param_cov,x_continuous = x_continuous, R_square = R_square))

    if len(plot_list) == 0:
        return None
    
    if export_param_statics is not None:
        table_element = "& " * (len(param)+2) +"\\\\ \hline \n"
        
    
    for plot_element, fmt in zip(plot_list,fmts):
        x, y, fmt, x_err, y_err, label, param, x_continuous = plot_element.get_coef()
       
        ax.errorbar(x,y,fmt=fmt,xerr = x_err,yerr = y_err, label =label)
        if param is not None:
            
            ax.plot(x_continuous,fitting_function(x_continuous,*param),'r-')
        
        if export_param_statics is not None:
            table_element = table_element + plot_element.get_param_statics()
    
    if export_param_statics is not None:
        with open(export_param_statics,'w') as f:
            f.write(table_element)
        
        
    
    if labels is not None:
        ax.legend()
    
    
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
            

    fig.tight_layout()
    return fig


def dictionary_boolean(
    dic1, dic2
):
    #return True while one dictionary fully included in the others
    #return False if one dictionary value is differecnt with the same key.
    #if the key value doesn't occurs in the larger dictionary, it returns KeyError
    
    if len(dic1) < len(dic2):
        dic2, dic1 = zip(dic1,dic2)

    
    for key in list(dic2.keys()):
        try:
            if dic1[key] != dic2[key]:
                return False
        except KeyError:
            print("Unexpected key from dict boolean")
            return KeyError
    
    return True

def get_regression_result(
    x,y,fitting_function, p0, maxfev = 100000
):
    param, param_covariance=opt.curve_fit(
        fitting_function,
        x,
        y,
        p0,
        maxfev= maxfev
    )

    return param,param_covariance

