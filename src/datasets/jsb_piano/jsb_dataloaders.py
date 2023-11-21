"""
JSB Piano Dataloaders
"""
from typing import Union, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .jsb_dataset import JSBPiano


def collate_fn(
    batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Collate function for JSB Piano dataset

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
        padded_data = pad_sequence(data, batch_first=True)
        padded_times = pad_sequence(times, batch_first=True)

        # Creating a mask
        mask = torch.zeros_like(padded_data, dtype=torch.bool)
        for i, seq in enumerate(data):
            seq_length = len(seq)
            mask[i, :seq_length] = 1

        return [padded_data, mask, padded_times]
    else:
        # Creating a mask
        padded_data = pad_sequence(batch, batch_first=True)
        mask = torch.zeros_like(padded_data, dtype=torch.bool)
        for i, seq in enumerate(batch):
            seq_length = len(seq)
            mask[i, :seq_length] = 1
        return [padded_data, mask]


def jsb_piano(
    subseq_len: int, batch_size: int, return_times: bool = False
) -> Dict[str, DataLoader]:
    """
    Get dataloaders for JSB Piano dataset

    Parameters
    ----------
    subseq_len : int
        Length of subsequences to use
    batch_size : int
        Batch size
    return_times : bool, by default False
        If True, return times as well as data

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for "train", "val", and "test" sets
    """
    train_dataset = JSBPiano("train", subseq_len, return_times)
    val_dataset = JSBPiano("val", subseq_len, return_times)
    test_dataset = JSBPiano("test", subseq_len, return_times)

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloaders["val"] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    dataloaders["test"] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return dataloaders
