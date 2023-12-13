# Referred to https://github.com/alistairewj/challenge2012/blob/master/prepare-data.ipynb for inspiration.

import os
import pandas as pd
import numpy as np
import pickle

# See details about the dataset here: https://physionet.org/content/challenge-2012/1.0.0/
physionet_path = '../Datasets/physionet.org/files/challenge-2012/1.0.0/'

# These are the general descriptors. The rest of the parameters are time series variables.
general_descriptors = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

time_series_params = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
    'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP',
    'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2',
    'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]

# Sometimes a patient has multiple entries for the same parameter at the same timestamp. 
# Depending on the parameter, we may want to take the mean or sum or max of these values 
# for that timestamp.
# If the parameter is not in the aggregation_functions dict, then we take the mean of the values.
aggregation_functions = {
    'Urine': 'sum', # Add all the urine outputs for that timestamp
    'MechVent': 'max', # True or False
}

def replace_value(df, param, below=None, above=None, new_val=np.nan):
    idx = df['Parameter'] == param

    # Don't replace values that are already -1.
    idx = idx & (df['Value'] != -1)

    if below is not None:
        idx = idx & (df['Value'] < below) 

    if above is not None:
        idx = idx & (df['Value'] > above)

    if 'function'in str(type(new_val)):
        df.loc[idx, 'Value'] = new_val(df.loc[idx, 'Value'])
    else: 
        df.loc[idx, 'Value'] = new_val
    return df


def replace_invalid_values(df):
    # Temp (°C)
    # Convert value from °F to °C.
    df = replace_value(df, 'Temp', below=113, above=82, new_val=lambda x: (x-32)*5/9)
    # If °C value is between 25 and 45, then keep it. Otherwise, replace with -1.
    df = replace_value(df, 'Temp', below=25)
    df = replace_value(df, 'Temp', above=45)
    
    # Heart rate (bpm)
    df = replace_value(df, 'HR', below=1)
    df = replace_value(df, 'HR', above=299)
    
    # Age (years)
    df = replace_value(df, 'Age', above=130)

    # Weight (kg)
    df = replace_value(df, 'Weight', below=35)
    df = replace_value(df, 'Weight', above=299)

    return df



dfs = []

for f in os.listdir(physionet_path + 'set-a/'):
    if not f.endswith('.txt'):
        continue

    # Read all time series data for one patient.    
    patient_df = pd.read_csv(physionet_path + 'set-a/' + f)
    dfs.append(patient_df)

for f in os.listdir(physionet_path + 'set-b/'):
    if not f.endswith('.txt'):
        continue

    # Read all time series data for one patient.    
    patient_df = pd.read_csv(physionet_path + 'set-b/' + f)
    dfs.append(patient_df)

patient_dfs = []
for patient_df in dfs:
    # Replace invalid values: 
    patient_df = replace_invalid_values(patient_df)

    # A patient can have multiple entries for the same parameter at the same timestamp. 
    # So, aggregate the values for each parameter at each timestamp using the corresponding aggregation
    # function for each parameter value.
    dfs = []
    for param_type in patient_df['Parameter'].unique():
        agg_func = aggregation_functions.get(param_type, 'mean')
        df = patient_df[patient_df['Parameter'] == param_type].groupby(['Time', 'Parameter']).agg(agg_func).reset_index()
        dfs.append(df)
    patient_df = pd.concat(dfs)

    # Convert time to an integer representing minutes.
    patient_df['Time'] = patient_df['Time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

    # Pivot the table so that each parameter is a column and each row is a timestamp.
    patient_df = patient_df.pivot(index='Time', columns='Parameter', values='Value').reset_index()

    # Ensure all columns are present
    for param in general_descriptors:
        if param not in patient_df.columns:
            patient_df[param] = np.nan
    for param in time_series_params:
        if param not in patient_df.columns:
            patient_df[param] = np.nan

    # Reorder columns
    patient_df = patient_df[['Time'] + general_descriptors + time_series_params]

    # Replace -1 with NaN
    patient_df = patient_df.replace(-1, np.nan)

    # Add the patient data frame to the list of patient dataframes.
    patient_df = patient_df.astype(float)
    patient_dfs.append(patient_df)
    

# Write the list of patient dataframes to a .pkl file
with open('../Datasets/physionet.pkl', 'wb') as f:
    pickle.dump(patient_dfs, f)

