"""
TODO: Add docstring
"""

import logging

import nni.nas.evaluator.pytorch.lightning as pl
import nni.nas.strategy as nas_strategy

from DeepCrazyhouse.src.domain.neural_net.nas.search_space.a0_nbrn_space import AlphaZeroSearchSpace
from DeepCrazyhouse.src.domain.neural_net.nas.evaluator.evaluators import OneShotChessModule
from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.configs.model_config import ModelConfig
from DeepCrazyhouse.src.domain.neural_net.nas.providers.dataset_provider import get_dataset

def get_search_space_from_args(name: str, mc: ModelConfig):
    r"""
    Returns the search space from the given input string. 

    :param name: Name of the search space to be returned.
    """
    logging.info(f"Setting search space \"{name}\"")

    if name == 'a0_nbrn':
        search_space = AlphaZeroSearchSpace()
    else:
        raise ValueError(f"Search space {name} not found.")

    return search_space
    

def get_evaluator_from_args(args, tc: TrainConfig):
    r"""
    Returns the evaluator method for the given arguments and train config.

    NOTE: The evaluator for multi-trial strategies is not implemented yet.
     
    If the search strategy is a one-shot strategy, the method constructs the evaluator from the search space, the trainer and the dataloaders and wraps them inside a (nni-traced) pytorch-lightning Lightning object.Here, both the search space and the trainer are constructed from the given arguments. The dataloaders for the training and validation dataset are constructed from the given train config.

    :param args: Arguments given to the command line tool.
    :param tc: TrainConfig
    :return: Lightning if the search strategy is a one-shot strategy, otherwise None (TODO)
    """
    category = get_category_from_strategy(args.search_strategy)
    verbose = args.debug
    logging.info(f"Setting evaluator \"{category}\"")

    if category == 'multi_trial':
        # TODO: Implement multi_trial evaluator
        raise NotImplementedError("Multi trial evaluator not implemented yet.")
    elif category == 'one_shot':
        module = OneShotChessModule(args=args, tc=tc, allow_teardown=False)
        trainer = get_lightning_trainer(args=args, tc=tc)
        train_dataloader = get_train_dataloader(tc=tc, verbose=verbose)
        val_dataloader = get_val_dataloader(tc=tc, verbose=verbose)

        return pl.Lightning(
            module,
            trainer,
            train_dataloader,
            val_dataloader
        )
    else:
        raise ValueError(f"Category {category} not found.")

def get_search_strategy_from_args(name: str):
    r"""
    Returns the search strategy from the given input string.
    For more information, please refer to https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html for more information.

    :param name: Name of the search strategy to be returned.
    """
    logging.info(f"Setting search strategy \"{name}\"")

    if name == 'random':
        search_strategy = nas_strategy.Random()
    elif name == 'grid':
        search_strategy = nas_strategy.GridSearch()
    elif name == 'evolution':
        search_strategy = nas_strategy.RegularizedEvolution()
    elif name == 'tpe':
        search_strategy = nas_strategy.TPE()
    elif name == 'pbrl':
        search_strategy = nas_strategy.PolicyBasedRL()
    elif name == 'darts':
        search_strategy = nas_strategy.DARTS()
    elif name == 'enas':
        search_strategy = nas_strategy.ENAS()
    elif name == 'gumbeldarts':
        search_strategy = nas_strategy.GumbelDARTS()
    elif name == 'random_one_shot':
        search_strategy = nas_strategy.RandomOneShot()
    elif name == 'proxyless':
        search_strategy = nas_strategy.Proxyless()
    else:
        raise ValueError(f"Search strategy {name} not found. Please refer to https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html for more information.")

    return search_strategy

def get_category_from_strategy(name: str):
    r"""
    Returns the category of the given search strategy.

    :param name: Name of the search strategy.
    """
    if name in ['random', 'grid', 'evolution', 'tpe', 'pbrl']:
        category = 'multi_trial'
    elif name in ['darts', 'enas', 'gumbeldarts', 'random_one_shot', 'proxyless']:
        category = 'one_shot'
    else:
        raise ValueError(f"Search strategy {name} not found. Please refer to https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html for more information.")

    return category

def get_lightning_trainer(args, tc: TrainConfig):
    r"""
    Returns the lightning trainer used for the neural architecture search.

    :return: Lightning trainer
    """
    return pl.Trainer(
        accelerator='gpu', 
        enable_progress_bar = True,
        devices = args.devices, # NOTE: Really important to set this to the correct devices! Otherwise, the trainer will (try to) use ALL available devices.
        limit_train_batches = tc.batch_steps if args.debug else 1.0,
        max_epochs = tc.nb_training_epochs,
    ) 

def get_train_dataloader(tc: TrainConfig, verbose: bool = True):
    r"""
    Returns the train dataloader. 
    
    The dataset is of type ``torch.utils.data.ConcatDataset``. All parts of the dataset from the folder are loaded and concatenated. Might need some adjustments for even larger datasets. (TODO)

    :param tc: TrainConfig
    :return: DataLoader
    """
    train_dataset = get_dataset(tc=tc, dataset_type="train", normalize=tc.normalize, verbose=verbose)

    return pl.DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=6)

def get_val_dataloader(tc: TrainConfig, verbose: bool = True):
    r"""
    Returns the validation dataloader.

    Currently only one part of the dataset is loaded. (TODO)

    :param tc: TrainConfig
    :return: DataLoader
    """
    val_dataset = get_dataset(tc=tc, dataset_type="val", normalize=tc.normalize, verbose=verbose)

    return pl.DataLoader(val_dataset, batch_size=tc.batch_size, num_workers=6)
