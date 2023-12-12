"""
@file: nas_config.py
Created on 02.12.23
@project: crazyara
@author: TimM4ster

Config definition file used for the neural architecture search. Appends necessary configurations to the main_config.py-file.
"""
from DeepCrazyhouse.configs.main_config import main_config

def get_nas_config():
    """
    Takes the main_config defined in DeepCrazyhouse/configs/main_config.py and appends the necessary configurations for the neural architecture search.
    """
    nc = main_config

    return nc

