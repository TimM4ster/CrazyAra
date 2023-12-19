import logging

import nni.nas.evaluator.pytorch.lightning as pl
import nni.nas.strategy as nas_strategy

from DeepCrazyhouse.src.domain.neural_net.nas.search_space.a0_nbrn_space import AlphaZeroSearchSpace
from DeepCrazyhouse.src.domain.neural_net.nas.evaluator.evaluators import OneShotModule
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent_pytorch import get_data_loader

def get_search_space_from_args(name: str):
    """
    Returns the search space from the given input string. 

    :param name: Name of the search space to be returned.
    """
    logging.info(f"Setting search space \"{name}\"")

    if name == 'a0_nbrn':
        search_space = AlphaZeroSearchSpace()
    else:
        raise ValueError(f"Search space {name} not found.")

    return search_space
    

def get_evaluator_from_args(name: str):
    """
    Returns the evaluator method from the given category of exploration strategies in the input string. 

    :param name: Name of the category of exploration strategies.
    """
    logging.info(f"Setting evaluator \"{name}\"")

    if name == 'multi_trial':
        # TODO: Implement multi_trial evaluator
        raise NotImplementedError("Multi trial evaluator not implemented yet.")
    elif name == 'one_shot':
        module = OneShotModule()
        trainer = get_lightning_trainer()
        train_dataloader = get_train_dataloader()
        val_dataloader = get_val_dataloader()

        return pl.Lightning(
            module,
            trainer,
            train_dataloader,
            val_dataloader
        )
    else:
        raise ValueError(f"Category {name} not found.")

def get_search_strategy_from_args(name: str):
    """
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

def get_lightning_trainer():
    """
    Returns the lightning trainer used for the neural architecture search.

    :return: Lightning trainer
    """
    return pl.Trainer(accelerator='gpu') # TODO: Test if this works. Potentially add training config.

def get_train_dataloader():
    """
    Returns the train dataloader.

    :return: DataLoader
    """
    _, x, y_value, y_policy, plys_to_end, _ = load_pgn_dataset(
        "train", 
        0, 
        True, 
        ... # TODO: Check training config for normalize
    )

    loader = get_data_loader(
        x, 
        y_value,
        y_policy,
        plys_to_end,
        ... # TODO: Add training config
    )

    return loader

def get_val_dataloader():
    """
    Returns the validation dataloader.

    :return: DataLoader
    """
    _, x, y_value, y_policy, plys_to_end, _ = load_pgn_dataset(
        "val", 
        0, 
        True, 
        ... # TODO: Check training config for normalize
    )

    loader = get_data_loader(
        x, 
        y_value,
        y_policy,
        plys_to_end,
        ... # TODO: Add training config
    )

    return loader
