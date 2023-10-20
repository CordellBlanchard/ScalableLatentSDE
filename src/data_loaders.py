from torch.utils.data import Dataset
import numpy as np

class SyntheticDatasetLoader(Dataset):
    '''
    CustomDataset for loading data.
    '''

    def __init__(self, data:np.ndarray):
        '''
        Initialize dataset

        Parameters
        ----------
        data : np.ndarray
            Data to load.
        '''
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]