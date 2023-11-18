
from typing import Dict

from torch.utils.data import DataLoader

from .dataset import LinearSyntheticPCD

def linear_synthetic_pcd(
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_time_steps: int,
    batch_size: int,
    return_times: bool = False
) -> Dict[str, DataLoader]:
    train_dataset = LinearSyntheticPCD(n_train_samples, n_time_steps, return_times)
    val_dataset = LinearSyntheticPCD(n_val_samples, n_time_steps, return_times)
    test_dataset = LinearSyntheticPCD(n_test_samples, n_time_steps, return_times)

    datasets = {}
    datasets["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    datasets["val"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    datasets["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return datasets