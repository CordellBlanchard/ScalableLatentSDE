"""
Dataloader for the physionet dataset in physionet_dataset.py
"""
from typing import Union, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from .physionet_dataset import PhysionetDataset
from torch.nn.utils.rnn import pad_sequence

def physionet(**kwargs) -> Dict[str, DataLoader]:
    """
    Get Physionet dataloaders for train, val, and test sets
    """
    return get_physionet_dataloaders(PhysionetDataset, **kwargs)

def collate_fn(
    batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Collate function for Physionet dataset

    Parameters
    ----------
    batch : Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        Batch either contains (data, times) or just data

    Returns
    -------
    Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        Collated batch
    """
    if isinstance(batch[0], tuple):
        data, times = zip(*batch)
        data = pad_sequence(data, batch_first=True)
        times = pad_sequence(times, batch_first=True)
        return data, times
    else:
        return pad_sequence(batch, batch_first=True)


def get_physionet_dataloaders(
    dataset_class: Dataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    imputation_method: str,
    discretization_method: str,
    batch_size: int,
    time_in_data: bool = False,
    return_times: bool = False,
    missing_in_data: bool = False,
    return_missing_mask: bool = False,
) -> Dict[str, DataLoader]:
    """
    General function to get dataloaders for Physionet datasets

    Parameters
    ----------
    dataset_class : Dataset
        Dataset class to use
    train_frac: float
        Fraction of data to use for training
    val_frac: float
        Fraction of data to use for validation
    test_frac: float
        Fraction of data to use for testing
    imputation_method : str
        Imputation method to use
    discretization_method : str
        Discretization method to use
    time_in_data: bool
        If True, include times in the data
    return_times: bool
        If True, return times as a separate tensor
    missing_in_data: bool
        If True, include missingness mask in the data
    return_missing_mask: bool
        If True, return missingness mask as a separate tensor
    batch_size : int
        Batch size

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    physionet_dataset = dataset_class(imputation_method, discretization_method, time_in_data, return_times, missing_in_data, return_missing_mask)

    # Split physionet dataset into train, val, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(physionet_dataset, [train_frac, val_frac, test_frac])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}