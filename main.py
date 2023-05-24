import single_photon_interference as spi
import pandas as pd
import pickle
import warnings


warnings.filterwarnings(action='ignore')



raw_data_file_name = "./spi_raw_data.xlsx"
df = pd.read_excel(raw_data_file_name, sheet_name= None)
sheets = df.keys()
datum : dict

try:
    with open("datum.pkl","rb") as f:
        datum = pickle.load(f)

except FileNotFoundError:

    for sheet in sheets:
        
        
    with open("./datum.pkl", "wb") as f:
        pickle.dump(datum,f)
