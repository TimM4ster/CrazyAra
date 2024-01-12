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
from pathlib import Path

sys.path.insert(0, '../../../../../')

from nni.nas.experiment import NasExperiment

from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.nas_config import get_base_configs
from DeepCrazyhouse.src.domain.neural_net.nas.nni_search_cli_util import *


def parse_args():
    """Defines the command-line arguments for the nni search and parses them."""

    parser = argparse.ArgumentParser(
        description="Neural architecture search script for searching CNNs or Transformer networks."
                    "Additional configuration settings can be set at:"
                    "CrazyAra/configs/nas_config.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--search-strategy", 
        type=str,
        help="Defines the search strategy for the neural architecture search. Parsed argument should be one of the following:"
        "random, grid, evolution, tpe, pbrl, darts, enas, gumbeldarts, random_one_shot, proxyless. Other search strategies are not supported."
        "For more information please refer to documentation in README.",
        default="darts"
    )

    parser.add_argument(
        "--export-dir", 
        type=str, 
        help="The export directory of neural architecture search results. This includes the top models, the experiment logs and the experiments themselves. The argument parsed as the experiment name will be used as a sub-directory name.", 
        default="/root/nas/"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the nas experiment. This will be used as a sub-directory name in the export directory.",
        default="crazyara_nas"
    )

    parser.add_argument(
        "--devices",
        nargs='+',
        type=int,
        help="List of devices to use for the nas experiment. If no devices are provided, all available devices will be used.",
        default=[0]
    )

    # TODO: Fix visualization
    parser.add_argument(
        "--port",
        type=int,
        help="NOTE: Does not work for one-shot strategies yet. Port for the nas experiment visualization.",
        default=8080
    )

    parser.add_argument(
        "--debug",
        type=bool,
        help="Decides whether to run the nas experiment in debug mode. If enabled, the experiment will be more verbose, load only one training dataset and only feature a single epoch with 10 batches.",
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
        logging.info(f"Torch version {torch.__version__} available with {torch.cuda.device_count()} GPUs. Running experiment...")  
    else: # if no gpus are available, abort
        sys.exit(f"Torch version {torch.__version__} does not recognize GPUs. Aborting...") 

    # train and model configs
    tc, mc = get_base_configs(args)

    # set export directory
    tc.export_dir = args.export_dir + args.experiment_name + '/'
    Path(tc.export_dir).mkdir(parents=True, exist_ok=True)

    # if debug mode is enabled, only run a single epoch with 10 batches
    if args.debug:
        tc.nb_parts = 1
        tc.nb_training_epochs = 1
        tc.batch_steps = 10

    # get search space from args
    search_space = get_search_space_from_args(args.search_space, mc)

    # get evaluator from args
    evaluator = get_evaluator_from_args(args, tc)

    # get search strategy from args
    search_strategy = get_search_strategy_from_args(args.search_strategy)

    # create experiment with search space, evaluator, search strategy and config
    exp = NasExperiment(
        search_space,
        evaluator,
        search_strategy,
    )

    exp.config.experiment_working_directory = tc.export_dir#

    if args.debug:
        logging.info(f"Visualization on port {args.port}...")
        logging.debug(f"Experiment name: {args.experiment_name}")
        logging.debug(f"Training service: {exp.config.training_service}")
        logging.debug(f"Trial concurrency: {exp.config.trial_concurrency}")
        logging.debug(f"Trial gpu number: {exp.config.trial_gpu_number}")
        logging.debug(f"Tuner gpu indices: {exp.config.tuner_gpu_indices}")

    exp.run(port=args.port, debug=args.debug)

    # export top models
    export_top_models(args, tc, exp)

if __name__ == "__main__":
    main()