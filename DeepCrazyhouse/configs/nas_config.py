"""
@file: nas_config.py
Created on 02.12.23
@project: crazyara
@author: TimM4ster

Config definition file used for the neural architecture search. Appends necessary configurations to the main_config.py-file.
"""
import glob

from nni.nas.experiment import NasExperimentConfig
from nni.experiment.config.training_services.local import LocalConfig

from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.configs.model_config import ModelConfig
from DeepCrazyhouse.configs.main_config import main_config

def get_nas_config(args, model_space, evaluator, strategy):
    """
    Fills the nas_config with the necessary configurations. Modify this function to change the configurations for the neural architecture search. Also takes the command-line arguments as input.

    :param nas_config: The nas_config to be filled.
    """
    # TODO: Fill nas_config with necessary configurations
    nas_config = NasExperimentConfig.default(model_space=model_space, evaluator=evaluator, strategy=strategy)
    nas_config.experiment_name = args.experiment_name
    nas_config.training_service = LocalConfig(use_active_gpu=True, gpu_indices=args.devices)

    return nas_config

def get_base_configs(args):
    """
    Takes the main_config defined in DeepCrazyhouse/configs/main_config.py and appends the necessary configurations for the neural architecture search.
    """
    tc = TrainConfig()
    tc.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))

    mc = ModelConfig()

    return tc, mc