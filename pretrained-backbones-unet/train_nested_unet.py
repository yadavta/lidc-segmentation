from backbones_unet.model.unet import Unet
from backbones_unet.model.unet_2 import Unet2
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.utils.lidc_dataset import LIDCDataset
from backbones_unet.utils.lidc_dataset_2 import LIDCDataset2
from backbones_unet.model.losses import DiceLoss, DiceLoss2
from backbones_unet.utils.trainer import Trainer
from backbones_unet.utils.lidc_trainer import LIDCTrainer
from backbones_unet.utils.lidc_trainer_2 import LIDCTrainer2
from backbones_unet.networks.UNet_Nested import UNet_Nested
import torch

from torch.nn import BCEWithLogitsLoss
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pickle

# create a torch.utils.data.Dataset/DataLoader
train_img_path = '/homes/iws/kzhang20/cse493/LIDC/images/training' 
train_mask_path = '/homes/iws/kzhang20/cse493/LIDC/annotations/training'

val_img_path = '/homes/iws/kzhang20/cse493/LIDC/images/validation' 
val_mask_path = '/homes/iws/kzhang20/cse493/LIDC/annotations/validation'

# augmentations
train_transforms = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
)
normalize = T.Normalize(0.15107111, 0.16035697) # apply normalize only to image

train_dataset = LIDCDataset(train_img_path, mask_paths=train_mask_path, transforms=train_transforms, normalize=normalize)
val_dataset = LIDCDataset(val_img_path, mask_paths=val_mask_path, normalize=normalize)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = UNet_Nested()

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.AdamW(params, 1e-2) # IOU 0.6348, 500 epochs, DICE loss
optimizer = torch.optim.Adam(params, 3e-3)
# optimizer = torch.optim.SGD(params, lr=2e-4, momentum=0.9, nesterov=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3, T_mult=5)
# scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.9, cooldown=2)

trainer = LIDCTrainer2(
    model,                    # UNet model with pretrained backbone
    criterion=DiceLoss(),     # loss function for model convergence
    optimizer=optimizer,      # optimizer for regularization
    lr_scheduler=scheduler,
#     scaler=scaler,
    epochs=100,                 # number of epochs for model training
    save_dir = 'runs/nested_unet_2'
)

trainer.fit(train_loader, val_loader)