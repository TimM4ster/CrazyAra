import logging

import nni.nas.strategy as nas_strategy
from DeepCrazyhouse.src.domain.neural_net.nas.search_space.a0_nbrn_space import AlphaZeroSearchSpace
from DeepCrazyhouse.src.domain.neural_net.nas.evaluator import BaseEvaluator, DefaultEvaluator

def get_search_space_from_name(name: str):
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
    

def get_evaluator_from_name(name: str) -> BaseEvaluator:
    """
    Returns the evaluator method from the given input string. 

    :param name: Name of the evaluator to be returned.
    """
    logging.info(f"Setting evaluator \"{name}\"")

    if name == 'default_evaluator':
        evaluator = DefaultEvaluator()
    else:
        raise ValueError(f"Evaluator {name} not found.")

    return evaluator

def get_search_strategy_from_name(name: str):
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
