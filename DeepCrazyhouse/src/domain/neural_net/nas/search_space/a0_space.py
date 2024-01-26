"""
@file: a0_nbrn_space.py
Created on 27.11.23
@project: CrazyAra
@author: TimM4ster

Neural Architecture Search model space for AlphaZero-based neural networks. Note that this model space only affects the architecture of the neural network, not the hyperparameters.
"""
from collections import OrderedDict
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, ParametrizedModule, Repeat
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem, _ValueHead, _PolicyHead, process_value_policy_head
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NTB, NCB
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU6

class AlphaZeroSearchSpace(ModelSpace):
    """
    Search space for AlphaZero-based neural networks. This search space only features a layer choice for the residual blocks.
    """
    
    def __init__(
            self,
            board_height: int = 8,
            board_width: int = 8,
            channels : int = 256,
            act_type : str = "relu",
            use_se : bool = False,
            num_res_blocks : int = 19
    ):
        super(AlphaZeroSearchSpace, self).__init__()

        self.op_candidates = OrderedDict([
            ("res_block", ResidualBlock(channels, act_type, use_se)),
            ("nbrn", NestedBottleneckResidualBlock(channels, act_type, use_se)),
            ("mobile_conv", MobileConvolutionBlock(channels)),
            ("ntb", NTB(channels, channels)),
            ("ncb", NCB(channels, channels))
        ])

        self.latency_dict = {
            0: 0.30431270599365237,
            1: 1.0131144523620606,
            2: 0.4259181022644043,
            3: 2.566568851470947,
            4: 0.6679224967956542
        }

        res_blocks = Repeat(
            lambda index: 
            LayerChoice(
                self.op_candidates,
                label=f"res_block_{index}"
            ), 
            num_res_blocks
        )

        # TODO 3: Check whether *res_blocks works properly
        self.body = Sequential(
            _Stem(channels=channels, act_type=act_type, nb_input_channels=52),
            *res_blocks
        )

        self.value_head = _ValueHead(
            board_height,
            board_width,
            channels,
            channels_value_head=8,
            fc0=256,
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
            policy_channels=81, 
            n_labels=64992,
            act_type = "relu", 
            select_policy_from_plane=False
        )

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the body
        :param x: Input to the body
        :return: Value & Policy Output
        """
        x = self.body(x)
        return process_value_policy_head(x, self.value_head, self.policy_head, False, False)
    

class NestedBottleneckResidualBlock(Module):
    # TODO 1: DOCSTRING
    def __init__(self, channels : int = 256, act_type : str = "relu", use_se : bool = False):
        super(NestedBottleneckResidualBlock, self).__init__()

        reduction_block = ReductionOrExpantionBlock(
            reduce=True, 
            channels=channels, 
            act_type=act_type
        )

        expansion_block = ReductionOrExpantionBlock(
            reduce=False, 
            channels=channels, 
            act_type=act_type
        )

        intermediate_block = ResidualBlock(
            channels=int(channels / 2), 
            act_type=act_type
        )

        self.body = Sequential(
            reduction_block,
            intermediate_block,
            intermediate_block,
            expansion_block
        )

        self.act = get_act(act_type)
    
    def forward(self, x):
        return self.act(x + self.body(x))


class ReductionOrExpantionBlock(Module):
    """
    Definition of a 1x1 convolutional block used for reducing or expanding of the number of channels.
    If reduce is set to True, the number of channels is reduced by half, otherwise it is doubled.
    """
    def __init__(self, reduce: bool = True, channels: int = 256, act_type: str = "relu"):

        super(ReductionOrExpantionBlock, self).__init__()

        in_channels = channels if reduce else int(channels / 2)
        out_channels = int(channels / 2) if reduce else channels

        conv = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(1, 1), 
            padding=(0, 0),
            bias=False
        )

        norm = BatchNorm2d(num_features=out_channels)

        act = get_act(act_type)

        self.body = Sequential(
            conv,
            norm,
            act
        )

    def forward(self, x):
        return self.body(x)


class ResidualBlock(Module):
    # TODO 2: DOCSTRING
    def __init__(self, channels: int = 256, act_type: str = "relu", use_se: bool = False):
        super(ResidualBlock, self).__init__()

        conv = Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
            bias=False
        )

        norm = BatchNorm2d(num_features=channels)

        act = get_act(act_type)

        self.body = Sequential(
            conv,
            norm,
            act
        )

    def forward(self, x):
        out = self.body(x)
        return x + self.body(out)
    

class MobileConvolutionBlock(Module):
    # TODO 3: DOCSTRING

    def __init__(self, in_channels: int = 256, expand_ratio: int = 6, use_se: bool = False):
        super(MobileConvolutionBlock, self).__init__()

        hidden_channels = int(in_channels * expand_ratio)

        expansion = Sequential(
            Conv2d(
                in_channels=in_channels, 
                out_channels=hidden_channels, 
                kernel_size=(1, 1), 
                padding=(0, 0),
                bias=False
            ),
            BatchNorm2d(num_features=hidden_channels),
            ReLU6(inplace=True)
        )

        depthwise = Sequential(
            Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=(3, 3), 
                padding=(1, 1),
                groups=hidden_channels,
                bias=False
            ),
            BatchNorm2d(num_features=hidden_channels),
            ReLU6(inplace=True)
        )

        reduction = Sequential(
            Conv2d(
                in_channels=hidden_channels, 
                out_channels=in_channels, 
                kernel_size=(1, 1), 
                padding=(0, 0),
                bias=False
            ),
            BatchNorm2d(num_features=in_channels),
        )

        self.body = Sequential(
            expansion,
            depthwise,
            reduction
        ) 

    def forward(self, x):
        return x + self.body(x)
