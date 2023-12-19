import nni

from nni.nas.evaluator.pytorch.lightning import LightningModule

@nni.trace
class OneShotModule(LightningModule):
    """
    Default evaluator for one-shot based neural architecture search with nni. Superclass LightningModule is a wrapper for PyTorch's LightningModule (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and simply adds the model to the module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Simply calls the forward method of the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        This method is called during the training loop. For every given batch, the loss is calculated and returned.
        For logging epoch-level metrics, one can use self.log() method. Further, the metrics are automatically logged to TensorBoard.

        :param batch: batch of data
        :param batch_idx: index of the batch
        :return: loss
        """
        # TODO: Implement training step
        pass

    def validation_step(self, batch, batch_idx):
        """
        This method is called during the validation loop. For every batch of the validation data, the loss is calculated and logged to TensorBoard using self.log().
        """
        # TODO: Implement validation step
        pass

    def configure_optimizers(self):
        """
        Sets up the optimizer and the learning rate scheduler. The optimizer is returned and used by the trainer.
        """
        # TODO: Implement 
        pass

    def on_validation_epoch_end(self):
        """
        Method is called after every validation epoch. It is used to report the intermediate result to nni.
        """
        nni.report_intermediate_result(
            self.trainer.callback_metrics["loss"].item()
        )

    def teardown(self, stage=None):
        """
        Method is called after the training is finished. It is used to report the final result to nni.
        """
        if stage == "fit":
            nni.report_final_result(self.trainer.callback_metrics["loss"].item())