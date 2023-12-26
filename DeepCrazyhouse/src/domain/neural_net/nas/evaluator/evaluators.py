import nni

import torch.optim as optim
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from nni.nas.evaluator.pytorch.lightning import LightningModule

from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.src.training.trainer_agent_pytorch import SoftCrossEntropyLoss

@nni.trace
class OneShotChessModule(LightningModule):
    """
    Default evaluator for one-shot based neural architecture search with nni. Superclass LightningModule is a wrapper for PyTorch's LightningModule (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and simply adds the model to the module.
    """
    def __init__(self, tc: TrainConfig):
        super().__init__()
        self.tc = tc
        self.value_loss = MSELoss()
        if self.tc.sparse_policy_label:
            self.policy_loss = CrossEntropyLoss()
        else:
            self.policy_loss = SoftCrossEntropyLoss()

    def forward(self, x):
        """
        Simply calls the forward method of the model.
        """
        return self.model(x)
    
    def training_step(self, batch, _):
        """
        This method is called during the training loop. For every given batch, the loss is calculated and returned.
        For logging epoch-level metrics, one can use self.log() method. Further, the metrics are automatically logged to TensorBoard.

        More information can be found on https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop.

        :param batch: batch of data
        :param batch_idx: index of the batch
        :return: loss
        """
        # Info 1: Model is set to training mode by default
        # Info 2: Gradients are automatically set to zero after every batch
        # Info 3: loss is passed backwards automatically
        # Info 4: optimizer performs step automatically

        # Step 1: Get the data from the batch
        data, value_label, policy_label = batch

        if self.tc.sparse_policy_label:
            policy_label = policy_label.long()

        # Step 2: Forward pass
        value_out, policy_out = self.forward(data)
        value_out.view(-1)

        # Step 3: Calculate losses
        # Step 3.1: Calculate value loss
        value_loss = self.value_loss(value_out, value_label)
        self.log("value_loss", value_loss)

        # Step 3.2: Calculate policy loss
        policy_loss = self.policy_loss(policy_out, policy_label)
        self.log("policy_loss", policy_loss)

        # Step 3.3: Calculate combined loss
        loss = self.get_total_loss(value_loss, policy_loss)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        This method is called during the validation loop. For every batch of the validation data, the loss is calculated and logged to TensorBoard using self.log().

        More information can be found on https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop.
        
        :param batch: batch of data
        :param batch_idx: index of the batch
        """
        # Step 1: Get the data from the batch
        data, value_label, policy_label = batch

        # Step 2: Forward pass
        value_out, policy_out = self.forward(data)

        # Step 3: Calculate losses
        # Step 3.1: Calculate value loss
        value_loss = self.value_loss(value_out, value_label)
        self.log("val_value_loss", value_loss)

        # Step 3.2: Calculate policy loss
        policy_loss = self.policy_loss(policy_out, policy_label)
        self.log("val_policy_loss", policy_loss)

        # Step 3.3: Calculate combined loss
        loss = self.get_total_loss(value_loss, policy_loss)
        self.log("val_loss", loss)


    def configure_optimizers(self):
        """
        Sets up the optimizer and the learning rate scheduler. The optimizer is returned and used by the trainer.

        A list of PyTorch optimizer-algorithms can be found on https://pytorch.org/docs/stable/optim.html#algorithms.

        Concerning the learning rate scheduler, more information and alternatives can be found on https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate.
        """

        optimizer = self.get_optimizer(self.tc)

        lr_scheduler = self.get_lr_scheduler(optimizer, self.tc)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss", # TODO: Maybe change to val_loss?
                "interval": "step", # Update learning rate stepwise...
                "frequency": 1, # ...every step
                "strict": True, # Value in monitor should be available
            },
        }

    def on_validation_epoch_end(self):
        """
        Method is called after every validation epoch. It is used to report the intermediate result to nni.
        """
        nni.report_intermediate_result(
            self.trainer.callback_metrics["val_loss"].item()
        )

    def teardown(self, stage=None):
        """
        Method is called after the training is finished. It is used to report the final result to nni.
        """
        if stage == "fit":
            nni.report_final_result(self.trainer.callback_metrics["val_loss"].item())

    def get_optimizer(self, tc: TrainConfig):
        """
        Returns the optimizer used for the neural architecture search. Currently, the SGD optimizer is used.

        :param tc: TrainConfig
        :return: optimizer
        """
        return optim.SGD(
            self.model.parameters(),
            lr=tc.max_lr, # TODO: max_lr???
            momentum=0.9, # TODO: Assign in TrainConfig
            weight_decay=tc.wd,
        )
    
    def get_lr_scheduler(self, optimizer: optim.Optimizer, tc: TrainConfig):
        """
        Returns the learning rate scheduler used for the neural architecture search. Currently, the OneCycleLR scheduler is used.

        :param optimizer: optimizer
        :param tc: TrainConfig
        :return: learning rate scheduler
        """
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tc.max_lr,
            epochs=tc.nb_training_epochs,
            steps_per_epoch=tc.batch_steps,
            pct_start=0.3,
            anneal_strategy="linear",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=tc.div_factor,
            final_div_factor=100,
            last_epoch=-1,
        )
    
    def get_total_loss(self, value_loss, policy_loss):
        """
        Returns the total loss used for the neural architecture search. Currently, the total loss is calculated by adding the weighted value loss and the policy loss.

        :param value_loss: value loss
        :param policy_loss: policy loss
        :param tc: TrainConfig
        :return: total loss
        """
        return value_loss * self.tc.val_loss_factor + policy_loss * self.tc.policy_loss_factor