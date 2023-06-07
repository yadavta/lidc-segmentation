import warnings
import torch
import math
import sys
from tqdm import tqdm, trange
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex
import os
import torch.nn.functional as F
import pickle


class Evaluator:
    """
    Trainer class that eases the training of a PyTorch model.
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    epochs : int
        The total number of iterations of all the training 
        data in one cycle for training the model.
    scaler : torch.cuda.amp
        The parameter can be used to normalize PyTorch Tensors 
        using native functions more detail:
        https://pytorch.org/docs/stable/index.html.
    lr_scheduler : torch.optim.lr_scheduler
        A predefined framework that adjusts the learning rate 
        between epochs or iterations as the training progresses.
    Attributes
    ----------
    train_losses_ : torch.tensor
        It is a log of train losses for each epoch step.
    val_losses_ : torch.tensor
        It is a log of validation losses for each epoch step.
    """
    def __init__(
        self, 
        model,  
        device=None,
        save_dir=None,
    ):
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.binary_metric = BinaryJaccardIndex().to(self.device)
        self.multi_metric = MulticlassJaccardIndex(2).to(self.device)
        self.save_dir = save_dir
        
        print('Saving to ' + save_dir)
        
    def fit(self, val_loader):
        """
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        """
        # attributes  
        self._evaluate(val_loader)
        
    
    @torch.inference_mode()
    def _evaluate(self, data_loader):
        self.model.eval()
        with tqdm(data_loader, unit=" eval-batch", colour="green") as evaluation:
            for i, (images, labels) in enumerate(evaluation):
                evaluation.set_description(f"Validation")
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).to(self.device)
                
                preds = F.sigmoid(preds)
                self.binary_metric(preds, labels)
                self.multi_metric(preds, labels)
                
        print('IOU: ' + str(self.binary_metric.compute().item()))
        print('mIOU: ' + str(self.multi_metric.compute().item()))
        
        test_iou = {'IOU': self.binary_metric.compute().item(), 'mIOU': self.multi_metric.compute().item()}
        
        with open(os.path.join(self.save_dir, 'test_iou.pkl'), 'wb') as fp:
            pickle.dump(test_iou, fp)

    def _get_device(self, _device):
        if _device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {device}"
            print(msg)
            return device
        return _device