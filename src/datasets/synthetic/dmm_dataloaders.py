"""
Dataloader for the synthetic datasets in dmm_datasets.py
"""
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from .dmm_datasets import LinearGSSM, NonLinearGSSM


def linear_gssm(**kwargs) -> Dict[str, DataLoader]:
    """
    Get LinearGSSM dataloaders for train, val, and test sets

    Parameters
    ----------
    **kwargs:
        Parameters for the dataset
        Should contain n_train_samples, n_val_samples, n_test_samples, n_time_steps, batch_size

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    return get_gssm_dataloaders(LinearGSSM, **kwargs)


def nonlinear_gssm(**kwargs) -> Dict[str, DataLoader]:
    """
    Get NonLinearGSSM dataloaders for train, val, and test sets

    Parameters
    ----------
    **kwargs:
        Parameters for the dataset
        Should contain n_train_samples, n_val_samples, n_test_samples, n_time_steps, batch_size

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    return get_gssm_dataloaders(NonLinearGSSM, **kwargs)


def get_gssm_dataloaders(
    dataset_class: Dataset,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_time_steps: int,
    batch_size: int,
    return_latents: bool = False,
    return_times: bool = False,
) -> Dict[str, DataLoader]:
    """
    General function to get dataloaders for GSSM datasets

    Parameters
    ----------
    dataset_class : Dataset
        Dataset class to use
    n_train_samples : int
        Number of training samples
    n_val_samples : int
        Number of validation samples
    n_test_samples : int
        Number of test samples
    n_time_steps : int
        Number of time steps
    batch_size : int
        Batch size
    return_latents: bool = False
        If True, return latents as well with observations
    return_times: bool = False
        If True, return time steps as well with observations

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    train_dataset = dataset_class(
        n_train_samples, n_time_steps, return_latents, return_times
    )
    val_dataset = dataset_class(
        n_val_samples, n_time_steps, return_latents, return_times
    )
    test_dataset = dataset_class(
        n_test_samples, n_time_steps, return_latents, return_times
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
