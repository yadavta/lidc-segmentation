import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np
import os
import re
from PIL import Image
import glob


class LIDCDataset2(Dataset):
    def __init__(self, 
                 img_paths, 
                 mask_paths=None,
                 mode='binary',
                 normalize=None,
                 transforms=None,
                 deeplab=False, 
                 resize=None):
        """
        Example semantic segmentation Dataset class.
        Run once when instantiating the Dataset object.
        If you want to use it for binary semantic segmentation, 
        please select the mode as 'binary'. For multi-class, enter 'multi'.
        example_data/
            └── /images/
                    └── 0001.png
                    └── 0002.png
                    └── 0003.png
                    └── ...
                /masks/
                    └── 0001_mask.png
                    └── 0002_mask.png
                    └── 0003_mask.png
                    └── ...
        img_paths : str
            The file path indicating the main directory that contains only images.
        mask_paths : str, default=None
            The file path indicating the main directory that contains only 
            ground truth images.
        size : tuple, default=(256, 256)
            Enter the (width, height) values into a tuple for resizing the data.
        mode : str, default='binary'
            Choose how the DataSet object should generate data. 
            Enter 'binary' for binary masks.
        normalize : torchvision.transforms.Normalize, default=None
            Normalize a tensor image with mean and standard deviation. 
            This transform does not support PIL Image.
        """
        self.img_paths = self._get_file_dir(img_paths)
        self.mask_paths = self._get_file_dir(mask_paths) if mask_paths is not None else mask_paths
        self.mode = mode
        self.normalize = normalize
        self.transforms = transforms
        self.deeplab = deeplab
        self.resize = resize
        
    def __len__(self):
        """
        Returns the number of samples in our dataset.
        Returns
        -------    
        num_datas : int    
            Number of datas.
        """
        return len(self.img_paths)
    
    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at 
        the given index idx. Based on the index, it 
        identifies the image’s location on disk, 
        converts that to a tensor using read_image, 
        retrieves the corresponding label from the 
        ground truth data in self.mask_paths, calls the transform 
        functions on them (if applicable), and returns 
        the tensor image and corresponding label in a tuple.
        Returns
        -------   
        img, mask : torch.Tensor
            The transformed image and its corresponding 
            mask image. If the mask path is None, it 
            will only return the transformed image.
            output_shape_mask: (batch_size, 1, img_size, img_size)
            output_shape_img: (batch_size, 3, img_size, img_size)
        """
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path))
        
        if self.resize is not None:
            img = np.array(Image.open(img_path).convert('L').resize((512, 512)), dtype=np.uint8)
            mask = np.array(Image.open(mask_path).resize((512, 512)), dtype=np.uint8) 
        
        mask = self._binary_mask(mask)
        img = torch.ByteTensor(img)
        mask = torch.ByteTensor(mask)
        
        img_mask = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), 0)
        
        img_mask = img_mask.unsqueeze(1)
        
        if self.transforms is not None:
            img_mask_transform = self.transforms(img_mask)
            img_transform = img_mask_transform[0]
            mask_transform = img_mask_transform[1]
            mask_transform[mask_transform > 0] = 1
        else:
            img_transform = img.unsqueeze(0)
            mask_transform = mask.unsqueeze(0)
        if self.deeplab:
            img_transform = F.to_pil_image(img_transform).convert('RGB')
            img_transform = torch.Tensor(np.array(img_transform, dtype=np.uint8).transpose((2, 0, 1)))
        
        return img_transform.float(), mask_transform.float()

    def _multi_class_mask(self, mask):
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        return masks

    def _binary_mask(self, mask):
        mask[:, :][mask[:, :] >= 1] = 1
        mask[:, :][mask[:, :] < 1] = 0
#         mask = np.expand_dims(mask, axis=0)
        return mask

    def _get_file_dir(self, directory):
        """
        Returns files in the entered directory.
        Parameters
        ----------
        directory : string
            File path.
        Returns
        -------
        directories: list
            All files in the directory.
        """
        def atoi(text):
            return int(text) if text.isdigit() else text
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text)]

        for roots,dirs,files in os.walk(directory):               
            if files:
                directories = [roots + os.sep + file for file in  files]
                directories.sort(key=natural_keys)
        return directories

#         return glob.glob(os.path.join(directory, '*.png'))