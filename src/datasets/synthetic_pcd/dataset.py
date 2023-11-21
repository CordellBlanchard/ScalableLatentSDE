import numpy as np
import torch
from torch.utils.data import Dataset

from .pcd import load_synthetic_data_trt

class LinearSyntheticPCD(Dataset):

    def __init__(
        self,
        n_samples : int,
        n_time_steps: int,
        return_times: bool = False
    ):
        self.n_samples = n_samples
        self.n_time_steps = n_time_steps
        self.return_times = return_times
        self._data = get_linear_synthetic_pcd(n_samples, n_time_steps, return_times)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self._data[idx]

def get_linear_synthetic_pcd(
    n_samples: int,
    n_time_steps: int,
    return_times: bool = False
):
    raw_data = load_synthetic_data_trt(
        fold_span = [1],
        nsamples = {"data" : n_samples},
        n_time_steps = n_time_steps
    )[1]["data"]

    # Using the raw data (the output of the black box above), convert into datasets
    # consisting of pairs of observations and hidden states over time.
    dataset = []
    for x, a in zip(raw_data["x"], raw_data["a"]):
        observation = x
        hidden = a

        sample = [observation, hidden]
        if return_times:
            sample.append(np.arange(n_time_steps))
        sample = [torch.from_numpy(x).float() for x in sample]

        dataset.append(sample)

    return dataset