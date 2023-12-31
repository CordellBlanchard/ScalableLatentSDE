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

    def __init__(self, imputation_method: str = "missing", discretization_method : str = "none", 
                 time_in_data: bool = False, return_times: bool = False, missing_in_data: bool = False,
                 return_missing_mask: bool = False):

        self.imputation_method = imputation_method
        self.discretization_method = discretization_method
        self.return_times = return_times
        self.return_missing_mask = return_missing_mask

        data = self.load_data()
        print("Loaded Physionet data")

        times = []
        discs = []
        missing_mask = []
        imputed_discretized_data = []

        for df in tqdm.tqdm(data):
            # Discritize the data using self.discretization_method.
            discretized = self.discretize(df)

            # Remove the general descriptors other than weight
            general_descriptors = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType']
            discretized = discretized.drop(columns=general_descriptors)

            # Save times and drop them from the data.
            times.append(discretized['Time'])
            discretized = discretized.drop(columns=['Time'])
            discs.append(discretized)

            # Indicate which values are missing for the columns other than Time.
            # In the missing mask, 0 means missing and 1 means not missing.
            missing_mask.append(discretized.notna())
        
        print("Done discretizing")

        combined_dfs = pd.concat(discs, axis=0)
        mins = combined_dfs.min()
        maxs = combined_dfs.max()

        for df in tqdm.tqdm(discs):
            # Normalize the data to between 0 and 1. 
            normalized_discretized = (df - mins) / (maxs - mins)

            # Impute the missing values using self.imputation_method.
            imputed = self.impute_missing(normalized_discretized)
            imputed_discretized_data.append(imputed)
            
        print("Done normalizing and imputing Physionet data")

        # Convert to torch tensors.
        self.times = [torch.tensor(time.astype(float).values, dtype=torch.float32) for time in times]
        self.missing_mask = [torch.tensor(df.astype(float).values, dtype=torch.float32) for df in missing_mask]
        imputed_discretized_data = [torch.tensor(df.astype(float).values, dtype=torch.float32) for df in imputed_discretized_data]

        # Include the time in the data if specified. 
        if time_in_data:
            imputed_discretized_data = [torch.cat([time.unsqueeze(1), df], dim=1) for time, df in zip(self.times, imputed_discretized_data)]

        # Include the missing information in the data if specified.
        if missing_in_data:
            imputed_discretized_data = [torch.cat([df, missing_df], dim=1) for df, missing_df in zip(imputed_discretized_data, self.missing_mask)]
        
        # Pad the missing mask with 1s to make it the same size as the data.
        pad_width = imputed_discretized_data[0].shape[1] - self.missing_mask[0].shape[1]
        self.missing_mask = [torch.cat([torch.ones(df.shape[0], pad_width), df], dim=1) for df in self.missing_mask]
        self.data = imputed_discretized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.return_missing_mask and self.return_times:
            return self.data[idx], self.missing_mask[idx], self.times[idx]
        elif self.return_missing_mask:
            return self.data[idx], self.missing_mask[idx]
        elif self.return_times:
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
            # If all the values for a variable are nan, set the mean to 0.
            # Otherwise, compute the mean of all the values for each variable.
            means = df.mean()
            means.fillna(0, inplace=True)
            df = df.fillna(means)
        else:
            raise ValueError("Invalid imputation method.")

        return df
  
    def discretize(self, df):
        '''
        Discretize the data using the specified method.
        '''
        if self.discretization_method == "none":
            # Convert time to hours from an integer rerpesenting hours.
            df['Time'] = df['Time'].map(lambda x: float(x / 60))
        elif self.discretization_method.isnumeric():
            # Discretize the data with intervals of size self.discretization_method minutes.
            df['Time'] = pd.to_datetime(df['Time'], unit='m')
            df = df.set_index('Time')
            df = df.resample(f'{self.discretization_method}T').mean()
            df = df.reset_index()
            # Convert the time back to hours.
            df['Time'] = df['Time'].map(lambda x: float(x.timestamp() / 3600))
        else:
            raise ValueError("Invalid discretization method.")

        return df

        


   