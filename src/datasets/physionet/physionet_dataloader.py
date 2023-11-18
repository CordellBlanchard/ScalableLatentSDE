"""
Dataloader for the physionet dataset in physionet_dataset.py
"""
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from .physionet_dataset import PhysionetDataset

def physionet(**kwargs) -> Dict[str, DataLoader]:
    """
    Get Physionet dataloaders for train, val, and test sets
    """
    return get_physionet_dataloaders(PhysionetDataset, **kwargs)


def get_physionet_dataloaders(
    dataset_class: Dataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    imputation_method: str,
    discretization_method: str,
    batch_size: int
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
    batch_size : int
        Batch size

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    physionet_dataset = dataset_class(imputation_method, discretization_method)

    # Split physionet dataset into train, val, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(physionet_dataset, [train_frac, val_frac, test_frac])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}