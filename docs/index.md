## Quick Links

- [Paper](https://drive.google.com/file/d/1bTLZDiEBEa07ajSySLb1_KffAw6OFiU_/view?usp=sharing)
- [Poster](https://drive.google.com/file/d/10KH6hrYE4J8k_6ep9odKOyde56Urh13k/view?usp=sharing)
- [Code](https://github.com/kzhang-20/lidc-segmentation)

Our research paper is written in IEEE-CPVR style. Our poster is in standard 32x24 format and was presented to our professors and TAs. We received full marks on this project for UW CSE's graduate-level [deep learning course](https://courses.cs.washington.edu/courses/cse493g1/23sp/).

## Research Highlights

#### Motivation

#### Pipeline

We introduced the first open-source "plug-and-play" pipeline for the LIDC dataset, entirely in PyTorch. This contribution enables the research community to easily try out different approaches to the LIDC dataset (whether it be expanding the neural network architecure, utilizing different augmentations, training on with a new loss function,  or really anything supported by PyTorch) *without* having to deal with the confusing world that is medical imaging. 

> In essence, our pipeline abstracts away the medical part of problem task and lets researchers treat it as a normal deep learning task.

Under the hood, our workflow follows this training procedure:
 
1. Download the 125 GB [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).
2. Utilize [pylidc](https://pylidc.github.io) and [Pidicom](https://github.com/pydicom/pydicom) to extract 10,005 2D slices of 512 x 512 pixel arrays and convert them to PNG images.
3. For the detection task, store each nodule's bounding box cooordinates in [MS-COCO format](https://cocodataset.org/#format-data).
4. For the segmentation task, crop each nodule image and mask with a 64 x 64 pixel bounding box.
5. Extract regions of interest (ROIs) using either Faster R-CNN or YOLO (our pipeline supports both).
6. Augment images using any of PyTorch's native augmentations. We also added support for RandAugment, AutoAugment, and TrivialAugment.
7. Utilize our custom-designed U-Net or another PyTorch-compatiable architecture for the LIDC task.

#### Augmentations

#### Comparison with Existing Work

#### Conclusion

## Contributors

- [Tanush Yadav](https://www.linkedin.com/in/tanushyadav/)
- [Kevin Zhang](mailto:kzhang20@cs.washington.edu)
- [Marius Rakickas](https://www.linkedin.com/in/marius-rakickas/)
