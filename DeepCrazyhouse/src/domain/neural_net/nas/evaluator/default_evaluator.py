import logging
import nni
from abc import ABC, abstractmethod

from nni.nas.nn.pytorch import ModelSpace

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate_model(model: ModelSpace):
        """
        This method acts as an evaluator for a given nni nas model. Calls both train_model and test_model methods.

        :param model: The model to evaluate
        """
        pass
    
    @abstractmethod
    def train_model(model: ModelSpace):
        """
        This method trains the given model.
        
        :param model: The model to train
        """
        pass
    
    @abstractmethod
    def test_model(model: ModelSpace):
        """
        Tests the given model in the current epoch. 

        :param model: The model to test
        """
        pass
    

class DefaultEvaluator(BaseEvaluator):

    def evaluate_model(model: ModelSpace):
        """
        This method evaluates the given model. As such, it iterates over all epochs (number is set in training config) and calls the train_model and test_model methods in every iteration. 
        The result of the test_model method is written to the nni intermediate report. After training, the best model is exported to the export directory.
        After the training is finished, the model architecture (including final weights) is exported as an onnx file to the export directory.

        :param model: The model to evaluate
        """

        # After training of model is finished, report final result to nni
        nni.report_final_result()


    def train_model(model: ModelSpace, epoch: int):
        """
        This method trains the given model for one epoch. Model is trained using the training agent.
        """
        logging.info(f"Training model in epoch {epoch}")

        pass

    def test_model(model: ModelSpace, epoch: int):
        """
        Tests the given model in the current epoch. Returns the value / policy loss defined in AlphaZero paper: 

        loss = (z - v)^2 - pi^T * log(p) + c * ||theta||^2
        """
        logging.info(f"Testing model in epoch {epoch}")

        # Steps:
        # 1. Get validation data
        # 2. Get output of model for validation data
        # 3. Compute loss

        data = ...

        # get actual value and policy for validation data
        v_val_out, v_policy_out = ...

        # get output of model for validation data
        m_val_out, m_policy_out = model(data)

        pass