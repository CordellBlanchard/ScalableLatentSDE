from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle

class PhysionetDataset(Dataset):
    '''
    Dataset for Physionet data.
    '''

    def __init__(self, imputation_method: str = "missing", discretization_method : str = "none"):

        self.imputation_method = imputation_method
        self.discretization_method = discretization_method

        data = self.load_data()
        imputed_discretized_data = []
        for df in data:
            discretized = self.discretize(df)
            imputed_discretized_data.append(self.impute_missing(discretized))

        # Make sure all the dataframes have the same number of rows.
        # Pad with rows of -1 if necessary.
        resized_data = []
        max_len = max([len(df) for df in imputed_discretized_data])
        for df in imputed_discretized_data:
            if len(df) < max_len:
                df = pd.concat([df, pd.DataFrame(-1, index=np.arange(len(df), max_len), columns=df.columns)])
            resized_data.append(df)

        self.data = resized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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
            # Otherwise, compute the mean of all the non -1 values for each variable.
            means = df[df != -1].mean()
            means.fillna(-1, inplace=True)
            df = df.fillna(means)
        else:
            raise ValueError("Invalid imputation method.")

        return df

    def mean_without_missing(self, series):
        '''
        Compute the mean of a series, ignoring the -1 values.
        '''
        included_vals = series[series != -1]
        if len(included_vals) == 0:
            return np.nan
        return included_vals.mean()
  
    def discretize(self, df):
        '''
        Discretize the data using the specified method.
        '''
        if self.discretization_method == "none":
            pass
        elif self.discretization_method.isnumeric():
            # Discretize the data with intervals of size self.discretization_method minutes.
            df['Time'] = pd.to_datetime(df['Time'], unit='m')
            df = df.set_index('Time')
            agg_funcs = {
                # if contains '_missing', then take the max of the column
                # otherwise, take the mean of the column
                col: 'max' if '_missing' in col else self.mean_without_missing for col in df.columns
            }
            df = df.resample(f'{self.discretization_method}T').agg(agg_funcs)
            df = df.reset_index()
        else:
            raise ValueError("Invalid discretization method.")

        return df

        


   