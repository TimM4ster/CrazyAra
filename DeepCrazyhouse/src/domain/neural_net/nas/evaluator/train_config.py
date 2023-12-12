from DeepCrazyhouse.configs.train_config import TrainConfig

def get_training_config():
    """
    Returns the training configuration. Settings are set in the set_settings method.
    """
    tc = TrainConfig()

    set_settings(tc)

    return tc


def set_settings(train_config: TrainConfig):
    """
    Sets the settings for the training process.
    """

    train_config.framework = "pytorch"

    # training context (cpu or gpu)
    train_config.context = "gpu"

    # device id
    train_config.device_id = 0

    # seed
    train_config.seed = 42

    train_config.export_weights = False
    train_config.log_metrics_to_tensorboard = False
    train_config.export_grad_histograms = False

    # directory to save the model to
    train_config.export_dir = "./models/"

    train_config.div_factor = 1

    train_config.batch_steps = 1000 * train_config.div_factor

    train_config.k_steps_initial = 0

    train_config.symbol_file = None
    train_config.params_file = None

    train_config.batch_size = int(1024 / train_config.div_factor)

    train_config.optimizer_name = "nag"

    train_config.max_lr = 0.07 / train_config.div_factor
    train_config.min_lr = 0.00001

    train_config.max_momentum = 0.95
    train_config.min_momentum = 0.8

    train_config.use_spike_recovery = True

    train_config.max_spikes = 20

    train_config.spike_thresh = 1.5

    train_config.wd = 1e-4

    train_config.dropout_rate = 0

    train_config.use_wdl = True

    train_config.use_plys_to_end = True

    train_config.use_mlp_wdl_ply = False

    train_config.val_loss_factor = 0.01

    train_config.policy_loss_factor = 0.988 if train_config.use_plys_to_end else 0.99

    train_config.plys_to_end_loss_factor = 0.002

    train_config.wdl_loss_factor = 0.01

    train_config.discount = 1.0

    train_config.normalize = True

    train_config.nb_training_epochs = 7

    train_config.select_policy_from_plane = True

    train_config.q_value_ratio = 0

    train_config.sparse_policy_label = True

    train_config.is_policy_from_plane_data = False

    train_config.name_initials = "TK"
