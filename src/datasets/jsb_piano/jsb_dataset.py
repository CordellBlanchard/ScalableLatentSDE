"""
JSB Chorales Dataset
Downloads the dataset by itself if it doesn't exist
"""
import os
import pickle
from typing import List, Union, Tuple

import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

DATASET_URL = (
    "https://github.com/czhuang/JSB-Chorales-dataset/raw/master/jsb-chorales-16th.pkl"
)


def download_file(url: str) -> Tuple[bool, str]:
    """
    Download a file from a given URL and save it to the same directory as this script.
    If the file already exists, the download is skipped.

    :param url: URL of the file to be downloaded.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Extract filename from the URL
    filename = url.split("/")[-1]

    # Create the full path for the file to be saved
    destination_path = os.path.join(script_dir, filename)

    # Check if the file already exists
    if os.path.exists(destination_path):
        return True, destination_path
    else:
        print("Downloading file...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
            print(f"File downloaded successfully: {destination_path}")
            return True, destination_path
        else:
            print("Failed to download the file.")
            return False, ""


class JSBPiano(Dataset):
    """
    JSB Chorales Dataset

    Parameters
    ----------
    split: str, by default "train"
        Which split to use, one of "train", "val", or "test"
    subseq_len: int, by default 64
        Length of subsequences to use
    return_times: bool, by default False
        If True, return times as well as data
    """

    def __init__(
        self, split: str = "train", subseq_len: int = 64, return_times: bool = False
    ):
        self.return_times = return_times
        self.subseq_len = subseq_len

        # Download dataset if it doesn't exist
        is_success, file_path = download_file(DATASET_URL)
        if not is_success:
            raise RuntimeError("Failed to download the dataset.")

        with open(file_path, "rb") as p:
            data = pickle.load(p, encoding="latin1")

        if split == "val":
            split = "valid"

        data = [self.encode_sequence(seq) for seq in data[split]]
        if return_times:
            times = [np.arange(len(seq)) for seq in data]
            self.times = [torch.from_numpy(seq).float() for seq in times]

        self.data = [torch.from_numpy(seq).float() for seq in data]

    def encode_step(self, midi_signals: List[int]) -> np.array:
        """
        Encode a single step of the sequence

        Parameters
        ----------
        midi_signals : List[int]
            List of MIDI signals for this step

        Returns
        -------
        np.array
            One-hot encoding of the MIDI signals
        """
        vals = np.eye(88)[np.array(list(midi_signals), dtype=int) - 21]
        return np.clip(np.sum(vals, axis=0), 0, 1)

    def encode_sequence(self, sequence: List[List[int]]) -> np.array:
        """
        Encode a sequence of MIDI signals

        Parameters
        ----------
        sequence : List[List[int]]
            List of MIDI signals for each step of the sequence

        Returns
        -------
        np.array
            One-hot encoding of the MIDI signals
        """
        return np.array([self.encode_step(step) for step in sequence])

    def plot(self, idx: int) -> None:
        """
        Plot a single sequence

        Parameters
        ----------
        idx : int
            Index of the sequence to plot
        """
        plt.figure(figsize=(15, 8))
        plt.imshow(self[idx].T, aspect="auto", cmap="gray_r", interpolation="none")
        plt.xlabel("Time Steps")
        plt.ylabel("MIDI Notes (0-87)")
        plt.title("MIDI Note Presses Over Time")
        plt.colorbar(
            label="Note Pressed (1) or Not Pressed (0)", orientation="vertical"
        )
        plt.show()

    def __len__(self):
        return sum(len(seq) // self.subseq_len for seq in self.data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get a single subsequence from the dataset

        Parameters
        ----------
        idx : int
            Index of the subsequence to get

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            If return_times is True, return a tuple of (data, times)
            Otherwise, return just the data
        """
        # Calculate which sequence this index corresponds to
        seq_idx = 0
        while idx >= len(self.data[seq_idx]) // self.subseq_len:
            idx -= len(self.data[seq_idx]) // self.subseq_len
            seq_idx += 1

        # Calculate the start and end of the subsequence
        start = idx * self.subseq_len
        end = start + self.subseq_len

        # Extract the subsequence
        subseq = self.data[seq_idx][start:end]

        if self.return_times:
            times_subseq = self.times[seq_idx][start:end]
            return subseq, times_subseq

        return [subseq]
