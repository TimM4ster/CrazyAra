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
import torch
import logging
import pickle

sys.path.insert(0, '../../../../../')

from nni.nas.experiment import NasExperiment

from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.nas_config import get_base_configs, get_nas_config
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
        help="", 
        default="/root/nni-logs"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the nas experiment. Used for logging purposes.",
        default="crazyara_nas"
    )

    parser.add_argument(
        "--devices",
        nargs='+',
        type=int,
        help="Default: GPU 0. List of devices to use for the nas experiment. If no devices are provided, all available devices will be used.",
        default=[0]
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port for the nas experiment visualization.",
        default=8080
    )

    parser.add_argument( # TODO: Integrate fully
        "--debug",
        type=bool,
        help="Decides whether to run the nas experiment in debug mode. If enabled, the experiment will be more verbose.",
        default=False
    )

    return parser.parse_args()

def main():
    # get args
    args = parse_args()

    # enable color logging
    enable_color_logging()

    # check if gpus are available
    if torch.cuda.is_available():
        if args.debug:
            logging.debug(f"Torch version {torch.__version__} available with {torch.cuda.device_count()} GPUs. Running experiment...")  
    else: # if no gpus are available, abort
        sys.exit(f"Torch version {torch.__version__} does not recognize GPUs. Aborting...") 

    # train and model configs
    tc, mc = get_base_configs(args)

    # get search space from args
    search_space = get_search_space_from_args(args.search_space, mc)

    # get evaluator from args
    evaluator = get_evaluator_from_args(args, tc)

    # get search strategy from args
    search_strategy = get_search_strategy_from_args(args.search_strategy)

    # get nas config from args and search space, evaluator and search strategy
    nas_config = get_nas_config(args, search_space, evaluator, search_strategy)

    # create experiment with search space, evaluator, search strategy and config
    exp = NasExperiment(
        search_space,
        evaluator,
        search_strategy, 
        nas_config
    )

    exp.config.trial_concurrency = len(args.devices)
    exp.config.trial_gpu_number = 1
    exp.config.tuner_gpu_indices = args.devices

    if args.debug:
        logging.info(f"Visualization on port {args.port}...")

    exp.run(port=args.port, debug=args.debug)

    if args.debug:
        logging.info("Saving top models...")

    top_models = exp.export_top_models(3, formatter='dict')
    with open(args.export_dir + 'top_models.pkl', 'wb+') as f:
        pickle.dump(top_models, f)

if __name__ == "__main__":
    main()