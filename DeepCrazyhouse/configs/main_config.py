"""
@file: main_config.py
Created on 24.09.18
@project: crazyara
@author: queensgambit

Main Config definition file used for the project.
Adjust the paths accordingly.
"""


# define the default dir where the training data in plane representation is located
# e.g. for supervised learning default_dir = "/data/planes/"
default_dir = "/data/"

if default_dir[-1] != "/":
    default_dir = default_dir + "/"

sl_dir = default_dir + "SL/"
pgn_dir = default_dir + "pgn/"

main_config = {
    # Crazyhouse - LichessDB
    # The training directory includes games from the months:        2016-01 - 2018-07 (excluding 2018-04 and 2018-08)
    # The validation directory includes games from the month:       2018-04
    # The test directory includes games from the month:             2018-08
    # The mate_in_one directory includes games from the month:      2018-04

    # Chess - KingBaseLite2019
    # The training directory includes games from the months:        2000-01 - 2018-12 (excluding 2012-09 and 2017-05)
    # The validation directory includes games from the month:       2012-09
    # The test directory includes games from the month:             2017-05
    # The mate_in_one directory includes games from the month:      lichess_db_standard_rated_2015-08.pgn

    # The pgn directories contain all files which are converted to plane representation
    "pgn_train_dir": pgn_dir + "train/",
    "pgn_val_dir": pgn_dir + "val/",
    "pgn_test_dir": pgn_dir + "test/",
    "pgn_mate_in_one_dir": pgn_dir + "mate_in_one/",
    # The plane directories contain the plane representation of the converted board state
    #  (.zip files which have been compressed by  the python zarr library)
    "planes_train_dir": sl_dir + "train/",
    "planes_val_dir": sl_dir + "val/",
    "planes_test_dir": sl_dir + "test/",
    "planes_mate_in_one_dir": sl_dir + "mate_in_one/",

    # The rec directory contains the plane representation which are used in the training loop of the network
    # use the notebook create_rec_dataset to generate the .rec files:
    # (Unfortunately when trying to start training with the big dataset a memory overflow occurred.
    # therefore the old working solution was used to train the latest model by loading the dataset via batch files)
    #  "train.idx", "val.idx", "test.idx", "mate_in_one.idx", "train.rec", "val.rec", "test.rec", "mate_in_one.rec"
    "rec_dir": "/home/demo_user/datasets/lichess/Crazyhouse/rec/",
    # The architecture dir contains the architecture definition of the network in mxnet .symbol format
    # These directories are used for inference
    "model_architecture_dir": "/home/demo_user/models/Crazyhouse/symbol/",
    # the weight directory contains the of the network in mxnet .params format
    "model_weights_dir": "/home/demo_user/models/Crazyhouse/params/",

    # layer name of the value output layer (e.g. value_tanh0 for legacy crazyhouse networks and value_out for newer
    # networks)
    "value_output": "value_out",
    # layer name of the policy output layer without softmax applied (e.g. flatten0 for legacy crazyhouse networks
    # policy_out for newer networks)
    "policy_output": "policy_out",
    "auxiliary_output": "auxiliary_out",
    "wdl_output": "wdl_out",
    "plys_to_end_output": "plys_to_end_out",

    # Active mode for different input & output representations.
    # Each mode is only compatible with a certain network input-/output representation:
    # Available modes:  0: MODE_CRAZYHOUSE    (crazyhouse only mode, no 960) available versions [1, 2, 3]
    #                   1: MODE_LICHESS       (all available lichess variants) available versions [1, 2 (last_moves), 3 (last_moves+fx-features)]
    #                   2: MODE_CHESS         (chess only mode, with 960) available versions [1, 2, 3]
    "mode": 2,
    "version": 3,
}
