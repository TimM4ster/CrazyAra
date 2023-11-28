from torch.nn import Sequential, Conv2d, BatchNorm2d, Module
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem, _ValueHead, _PolicyHead, process_value_policy_head, get_se

class NestedBottleNeckResidualBlock(Module):
    """
    Definition of a nested bottleneck residual block.
    """

    def __init__(self, channels, act_type: str, use_se: bool = False):
        super(NestedBottleNeckResidualBlock, self).__init__()

        self.inConv = Conv2d(in_channels=channels, out_channels=int(channels/2), kernel_size=(1, 1), padding=(0, 0))

        self.outConv = Conv2d(in_channels=int(channels/2), out_channels=channels, kernel_size=(1, 1), padding=(0, 0))

        self.bnormhalfc = BatchNorm2d(num_features=int(channels/2))

        self.bnormc = BatchNorm2d(num_features=channels)

        self.act = get_act(act_type)

        self.intermediate_block = IntermediateResidualBlock(channels=int(channels/2), act_type=act_type)

        self.body = Sequential(
            self.inConv,
            self.bnormhalfc,
            self.act,
            self.intermediate_block,
            self.intermediate_block,
            self.outConv,
            self.bnormc,
            self.act
        )

        self.use_se = use_se

        if use_se:
            self.se = get_se(se_type="se", channels=channels, use_hard_sigmoid=False)

    def forward(self, x):
        residual = x

        out = self.body(x)

        if self.use_se:
            out = self.se(out)

        return self.act(out + residual)


class IntermediateResidualBlock(Module):
    """
    Definition of a residual block inside the nested bottleneck residual block. 
    """

    def __init__(self, channels, act_type: str):
        super(IntermediateResidualBlock, self).__init__()

        self.conv = Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1))

        self.bnorm = BatchNorm2d(num_features=channels)

        self.act = get_act(act_type)

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bnorm(out)
        out = self.act(out)

        out = self.conv(out)
        out = self.bnorm(out)

        return self.act(out + residual)


class AlphaZeroNBRN(Module):
    """
    Creates the alpha zero net with the nbrn architecture.
    """
    def __init__(
        self,
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
        super(AlphaZeroNBRN, self).__init__()
        self.use_plys_to_end = use_plys_to_end
        self.use_wdl = use_wdl

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(NestedBottleNeckResidualBlock(channels, act_type, use_se))

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
    

def get_nbrn_alpha_zero_model(args):
    return AlphaZeroNBRN(
        n_labels=args.n_labels,
        channels=args.channels,
        nb_input_channels=args.nb_input_channels,
        board_height=args.board_height,
        board_width=args.board_width,
        channels_value_head=args.channels_value_head,
        channels_policy_head=args.channels_policy_head,
        num_res_blocks=args.num_res_blocks,
        value_fc_size=args.value_fc_size,
        act_type=args.act_type,
        select_policy_from_plane=args.select_policy_from_plane,
        use_wdl=args.use_wdl,
        use_plys_to_end=args.use_plys_to_end,
        use_mlp_wdl_ply=args.use_mlp_wdl_ply,
        use_se=args.use_se,
    )