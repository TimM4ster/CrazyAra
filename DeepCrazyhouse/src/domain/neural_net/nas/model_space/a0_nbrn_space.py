"""
@file: a0_nbrn_space.py
Created on 27.11.23
@project: CrazyAra
@author: TimM4ster

Neural Architecture Search model space for A0-NBRN
"""
import nni
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, ParametrizedModule
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem, _ValueHead, _PolicyHead, process_value_policy_head, get_se
from torch.nn import Module, Sequential

class A0_NBRN_Model_Space(ModelSpace):
    
    def __init__(
            self,
            board_height: int = 8,
            board_width: int = 8,
            channels : int = 256,
    ):
        super(A0_NBRN_Model_Space, self).__init__()

        num_res_blocks = nni.choice(
            'num_res_blocks',
            [10, 19, 39]
        )

        res_blocks = []

        for i in range(num_res_blocks):
            res_blocks.append(
                LayerChoice(
                    [
                        NestedBottleneckResidualBlock(channels, "relu", False), 
                        BaseResidualBlock(channels, "relu", False)
                    ],
                    label=f"res_block_{i}"
                )
            )

        self.body = Sequential(
            _Stem(board_height, board_width, channels),
            *res_blocks
        )

        self.value_head = _ValueHead(
            board_height,
            board_width,
            channels,
            channels_value_head=8,
            value_fc_size=256,
            act_type = "relu",
            use_raw_features=False,
            nb_input_channels=52,
            use_wdl=False,
            use_plys_to_end=False,
            use_mlp_wdl_ply=False
        )

        self.policy_head = _PolicyHead(
            board_height, 
            board_width, 
            channels, 
            channels_policy_head=81, 
            n_labels=64992,
            act_type = "relu", 
            select_policy_from_plane=False
        )


class NestedBottleneckResidualBlock(ParametrizedModule):
    pass

class BaseResidualBlock(Module):
    pass