from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
import tqdm
import torch

class PhysionetDataset(Dataset):
    '''
    Dataset for Physionet data.
    '''

    def __init__(self, imputation_method: str = "missing", discretization_method : str = "none", time_in_data: bool = False, return_times: bool = False, missing_in_data: bool = False):

        self.imputation_method = imputation_method
        self.discretization_method = discretization_method
        self.return_times = return_times

        data = self.load_data()
        print("Loaded Physionet data")

        times = []
        general_descs = []
        missing_data = []
        imputed_discretized_data = []

        for df in tqdm.tqdm(data):
            # Discritize the data using self.discretization_method.
            discretized = self.discretize(df)
            general_descriptors = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
            # Use forward imputation for the general descriptors.
            discretized[general_descriptors] = discretized[general_descriptors].ffill()
            discretized[general_descriptors] = discretized[general_descriptors].fillna(-1)

            # Save times and general descriptors.
            times.append(discretized['Time'])
            general_descs.append(discretized[general_descriptors])

            # Indicate which values are missing for the columns that are not general descriptors or Time.
            discretized = discretized.drop(columns=general_descriptors + ['Time'])
            missing_data.append(discretized.isna())

            # Normalize the data except for the columns containing general descriptors, missing information and time.
            means = discretized.mean()
            stds = discretized.std()
            normalized_discretized = (discretized-means) / stds

            # Impute the missing values using self.imputation_method.
            imputed_discretized_data.append(self.impute_missing(normalized_discretized))
            
        print("Done discretizing and imputing Physionet data")

        # Concatenate the general descriptors and the imputed, discretized data.
        imputed_discretized_data = [pd.concat([general_desc, df], axis=1) for general_desc, df in zip(general_descs, imputed_discretized_data)]
        # Convert to torch tensors.
        self.times = [torch.tensor(time.astype(float).values, dtype=torch.float32) for time in times]
        missing_data = [torch.tensor(df.astype(float).values, dtype=torch.float32) for df in missing_data]
        imputed_discretized_data = [torch.tensor(df.astype(float).values, dtype=torch.float32) for df in imputed_discretized_data]

        # Include the time in the data if specified. 
        if time_in_data:
            imputed_discretized_data = [torch.cat([time.unsqueeze(1), df], dim=1) for time, df in zip(self.times, imputed_discretized_data)]

        # Include the missing information in the data if specified.
        if missing_in_data:
            imputed_discretized_data = [torch.cat([df, missing_df], dim=1) for df, missing_df in zip(imputed_discretized_data, missing_data)]

        self.data = imputed_discretized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.return_times:
            return self.data[idx], self.times[idx]
        else:
            return self.data[idx]
    
    def load_data(self):
        '''
        Load the data from the pkl file.
        '''
        with open("../Datasets/physionet.pkl", "rb") as f:
            data = pickle.load(f)

        return data
    
    def impute_missing(self, df):
        '''
        Impute missing values using the specified method.
        '''
        if self.imputation_method == "missing":
            df = df.fillna(-1)
        elif self.imputation_method == "forward":
            df = df.ffill()
            # Fill in NaNs before the first observation with -1.
            df = df.fillna(-1)
        elif self.imputation_method == "mean":
            # If all the values for a variable are -1, set the mean to -1.
            # Otherwise, compute the mean of all the values for each variable.
            means = df.mean()
            means.fillna(-1, inplace=True)
            df = df.fillna(means)
        else:
            raise ValueError("Invalid imputation method.")

        return df
  
    def discretize(self, df):
        '''
        Discretize the data using the specified method.
        '''
        if self.discretization_method == "none":
            # Convert time to hours from an integer rerpesenting minutes.
            df['Time'] = df['Time'].map(lambda x: int(x / 60))
        elif self.discretization_method.isnumeric():
            # Discretize the data with intervals of size self.discretization_method minutes.
            df['Time'] = pd.to_datetime(df['Time'], unit='m')
            df = df.set_index('Time')
            df = df.resample(f'{self.discretization_method}T').mean()
            df = df.reset_index()
            # Convert the time back to hours.
            df['Time'] = df['Time'].map(lambda x: int(x.timestamp() / 3600))
        else:
            raise ValueError("Invalid discretization method.")

        return df

        


   