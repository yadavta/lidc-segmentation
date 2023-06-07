from backbones_unet.model.unet import Unet
from backbones_unet.model.unet_2 import Unet2
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.utils.lidc_dataset import LIDCDataset
from backbones_unet.utils.lidc_dataset_2 import LIDCDataset2
from backbones_unet.model.losses import DiceLoss, DiceLoss2
from backbones_unet.utils.trainer import Trainer
from backbones_unet.utils.lidc_trainer import LIDCTrainer
from backbones_unet.utils.lidc_trainer_2 import LIDCTrainer2
from backbones_unet.utils.evaluator import Evaluator
import torch
import os

from torch.nn import BCEWithLogitsLoss
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pickle

val_img_path = '/homes/iws/kzhang20/cse493/LIDC/images/testing' 
val_mask_path = '/homes/iws/kzhang20/cse493/LIDC/annotations/testing'
save_dir = 'runs/unet_1/'

# normalize = T.Normalize(0.15107111, 0.16035697) # apply normalize only to image

val_dataset = LIDCDataset(val_img_path, val_mask_path)

val_loader = DataLoader(val_dataset, batch_size=64)

model = Unet2(num_classes=2)
model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt')))
evaluator = Evaluator(model, save_dir=save_dir)
evaluator.fit(val_loader)