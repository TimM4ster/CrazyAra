"""
@file: dataset_provider.py - subject to change
Created on 20.12.23
@project: CrazyAra
@author: TimM4ster

Provides the dataset for the neural architecture search.
"""
import logging

from torch.utils.data.dataset import ConcatDataset

from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.train_util import prepare_policy

def get_dataset(tc: TrainConfig, dataset_type: str = "train", normalize: bool = False, verbose: bool = True):
    """
    TODO
    """
    if dataset_type == "train":
        if verbose:
            logging.info("Loading training dataset...")

        return _get_train_dataset(tc=tc, size=tc.nb_parts, normalize=normalize, verbose=verbose)
    elif dataset_type == "val":
        logging.info("Loading validation dataset...")

        return _get_val_dataset(tc=tc, size=1, normalize=normalize, verbose=verbose)
    
    
def _get_train_dataset(tc: TrainConfig, size: int, normalize: bool = False, verbose: bool = True):
    """
    TODO
    """
    datasets = []
    for part_id in range(size):
        if verbose:
            logging.info(f"Loading training dataset part {part_id + 1} of {size}...")

        datasets.append(
            _get_tensor_dataset(
                tc=tc,
                dataset_type="train",
                part_id=part_id,
                normalize=normalize,
                verbose=verbose
            )
        )
    
    return ConcatDataset(datasets)

def _get_val_dataset(tc: TrainConfig, size: int, normalize: bool = False, verbose: bool = True):
    """
    TODO
    """
    return _get_tensor_dataset(tc=tc, dataset_type="val", part_id=0, normalize=normalize, verbose=verbose)

def _get_tensor_dataset(tc: TrainConfig, dataset_type: str, part_id: int = 0, normalize: bool = False, verbose: bool = True):
    """
    TODO
    """
    _, x, y_value, y_policy, _, _ = load_pgn_dataset(
        dataset_type=dataset_type,
        part_id=part_id,
        verbose=verbose,
        normalize=normalize
    )

    y_policy_prep = prepare_policy(
        y_policy, 
        tc.select_policy_from_plane, 
        tc.sparse_policy_label, 
        tc.is_policy_from_plane_data
    )

    return x, y_value, y_policy_prep