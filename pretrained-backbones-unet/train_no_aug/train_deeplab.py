from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer
from backbones_unet.utils.lidc_trainer import LIDCTrainer
from network._deeplab import DeepLabV3
import torch
import network
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import pickle

# create a torch.utils.data.Dataset/DataLoader
train_img_path = '/homes/iws/kzhang20/cse493/LIDC/images/training' 
train_mask_path = '/homes/iws/kzhang20/cse493/LIDC/annotations/training'

val_img_path = '/homes/iws/kzhang20/cse493/LIDC/images/validation' 
val_mask_path = '/homes/iws/kzhang20/cse493/LIDC/annotations/validation'

train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path)
val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path)

train_loader = DataLoader(train_dataset, batch_size=4)
val_loader = DataLoader(val_dataset, batch_size=4)

model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=1, output_stride=4)

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.AdamW(params, 3e-4) 
optimizer = torch.optim.AdamW(params, 3e-4)

trainer = LIDCTrainer(
    model,                    # UNet model with pretrained backbone
    criterion=DiceLoss(),     # loss function for model convergence
    optimizer=optimizer,      # optimizer for regularization
    epochs=100                 # number of epochs for model training
)

trainer.fit(train_loader, val_loader)
torch.save(model.state_dict(), 'runs/deeplab_resnet50_1/model.pt')

with open('runs/deeplab_resnet50_2/train_losses.pkl', 'wb') as fp:
    pickle.dump(trainer.train_losses_, fp)
with open('runs/deeplab_resnet50_2/val_iou.pkl', 'wb') as fp:
    pickle.dump(trainer.val_losses_, fp)