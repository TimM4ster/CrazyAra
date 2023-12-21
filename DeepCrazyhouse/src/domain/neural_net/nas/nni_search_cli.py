"""
@file: nni_search.py
Created on 17.12.23
@project: CrazyAra
@author: TimM4ster

Provides a command-line interface to run neural architecture search using nni. For the main configuration files please modify
* CrazyAra/DeepCrazyhouse/configs/main_config.py
* CrazyAra/DeepCrazyhouse/configs/train_config.py
* CrazyAra/DeepCrazyhouse/configs/nas_config.py
"""
import argparse
import sys

sys.path.insert(0, '../../../../../')

from nni.nas.experiment import NasExperiment

from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.nas_config import get_nas_configs
from DeepCrazyhouse.src.domain.neural_net.nas.nni_search_cli_util import *

def parse_args():
    """Defines the command-line arguments for the nni search and parses them."""

    parser = argparse.ArgumentParser(
        description="Neural architecture search script for searching CNNs or Transformer networks."
                    "Additional configuration settings can be set at:"
                    "CrazyAra/configs/nas_config.py"
    )

    parser.add_argument(
        "--search-space", 
        type=str,
        help="Defines the search space for the neural architecture search. Parsed argument should match name of file in:" 
        "CrazyAra/DeepCrazyhouse/src/domain/neural_net/nas/search_space/."
        "For more information please refer to documentation in README.",
        default="a0_nbrn"
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Defines the category of the neural architecture search, specifically the category of the exploration strategy. Currently, only the categories \"multi_trial\" and \"one_shot\" are supported. Note that the evaluator is selected accordingly."
        "For more information please refer to documentation in README.",
        default="one_shot"
    )

    parser.add_argument(
        "--search-strategy", 
        type=str,
        help="Defines the search strategy for the neural architecture search. Parsed argument should be one of the following:"
        "random, grid, evolution, tpe, pbrl, darts, enas, gumbeldarts, random_one_shot, proxyless. Other search strategies are not supported."
        "For more information please refer to documentation in README.",
        default="darts"
    )

    # TODO: Review and potentially remove
    parser.add_argument(
        "--name-initials", 
        type=str, 
        help="Name initials which are used to identify running training processes with the rtpt library", 
        default="XX"
    )

    # TODO: Add docstring
    parser.add_argument(
        "--export-dir", 
        type=str, 
        help="TODO: Add docstring", 
        default="./"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the nas experiment. Used for logging purposes.",
        default="crazyara_nas"
    )

    parser.add_argument(
        "--use-multi-gpu",
        type=bool,
        help="Decides whether to use multiple gpus for nas experiment.",
        default=False
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port for the nas experiment visualization.",
        default=8080
    )

    return parser.parse_args()

def main():
    # get args
    args = parse_args()

    # enable color logging
    enable_color_logging()

    # configs for nas experiment
    tc, mc, nc = get_nas_configs(args)

    # get search space from args
    search_space = get_search_space_from_args(args.search_space, mc)

    # get evaluator from args
    evaluator = get_evaluator_from_args(args.category, tc)

    # get search strategy from args
    search_strategy = get_search_strategy_from_args(args.search_strategy)

    # create experiment with search space, evaluator, search strategy and config
    exp = NasExperiment(
        search_space,
        evaluator,
        search_strategy, 
        #nas_config
    )

if __name__ == "__main__":
    main()