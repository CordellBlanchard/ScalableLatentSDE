
from typing import Dict

from .pcd import load_synthetic_data_trt

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def linear_synthetic_pcd(
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_time_steps: int,
    batch_size: int,
    return_times: bool = False
):
    raw_datasets = load_synthetic_data_trt(
        fold_span = [1],
        nsamples = {
            "train" : n_train_samples,
            "valid" : n_val_samples,
            "test"  : n_test_samples
        },
        n_time_steps = n_time_steps
    )[1]

    # Using the raw data (the output of the black box above), convert into datasets
    # consisting of pairs of observations and hidden states over time.
    datasets = {}
    for split, data in raw_datasets.items():
        dataset = []
        for x, x_orig, a in zip(data["x"], data["x_orig"], data["a"]):
            observation = x
            hidden = np.concatenate([x_orig, a], axis = -1)

            sample = [observation, hidden]
            if return_times:
                sample.append(np.arange(n_time_steps))
            sample = [torch.from_numpy(x) for x in sample]

            dataset.append(sample)

        datasets[split] = data

    datasets["train"] = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    datasets["valid"] = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False)
    datasets["test"] = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    
    # Rename
    datasets["val"] = datasets.pop("valid")
    return datasets