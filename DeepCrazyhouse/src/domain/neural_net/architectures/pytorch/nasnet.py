"""
@file: nasnet.py
Created on 23.01.23
@project: CrazyAra
@author: TimM4ster
"""
import os
import glob
import pickle
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module, ReLU6
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem, _ValueHead, _PolicyHead, process_value_policy_head
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NTB, NCB

class NASNet(Module):
    """
    Creates the alpha zero net with the nbrn architecture.
    """
    def __init__(
        self,
        model_path: str = None,
        n_labels=64992,
        channels=256,
        nb_input_channels=52,
        board_height=8,
        board_width=8,
        channels_value_head=8,
        channels_policy_head=81,
        num_res_blocks=19,
        value_fc_size=256,
        act_type="relu",
        select_policy_from_plane=False,
        use_wdl=False, use_plys_to_end=False,
        use_mlp_wdl_ply=False,
        use_se=False,
    ):
        """
        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param channels_policy_head: Number of channels in the bottle neck for the policy head
        :param channels_value_head: Number of channels in the bottle neck for the value head
        :param num_res_blocks: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :return: net description
        """
        super(NASNet, self).__init__()

        if model_path is None:
            raise ValueError("model_path must be specified")

        self.use_plys_to_end = use_plys_to_end
        self.use_wdl = use_wdl

        res_blocks = get_blocks_from_file(model_path)

        self.body = Sequential(_Stem(channels=channels, act_type=act_type,
                                     nb_input_channels=nb_input_channels),
                               *res_blocks)

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHead(board_height, board_width, channels, channels_value_head, value_fc_size,
                                     act_type, False, nb_input_channels,
                                     use_wdl, use_plys_to_end, use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                                       act_type, select_policy_from_plane)
        
    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)

        return process_value_policy_head(out, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)
    

def get_blocks_from_file(directory: str):
    """
    Loads the network architecture from the .pkl file in the given directory.
    :param directory: Path to the directory containing the .pkl file
    :return: List of Blocks
    """
    # Find the .pkl file in the directory
    file_path = glob.glob(os.path.join(directory, '*.pkl'))[0]

    with open(file_path, "rb") as f:
        network_architecture: dict = pickle.load(f)
    
    res_blocks = []
    for block in network_architecture.values():
        res_blocks.append(extract_block(block))

    return res_blocks

def extract_block(block: str):
    """
    Extracts a block from the given string.
    :param block: String representing the block
    :return: Block
    """
    channels = 256
    act_type = "relu"
    use_se = False

    if block == "res_block":
        return ResidualBlock(channels, act_type, use_se)
    elif block == "mobile_conv":
        return MobileConvolutionBlock(channels)
    elif block == "ncb":
        return NCB(channels, channels)
    elif block == "ntb":
        return NTB(channels, channels)
    elif block == "nbrn":   
        return NestedBottleneckResidualBlock(channels, act_type, use_se)
    else:
        raise ValueError(f"Unknown block type: {block}")

def get_nasnet_model(args, model_path: str):
    model = NASNet(model_path=model_path, channels=256, channels_value_head=4,
                            channels_policy_head=args.channels_policy_head,
                            value_fc_size=256, num_res_blocks=19, act_type='relu',
                            n_labels=args.n_labels, select_policy_from_plane=args.select_policy_from_plane,
                            use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end,
                            use_mlp_wdl_ply=args.use_mlp_wdl_ply)
    return model

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