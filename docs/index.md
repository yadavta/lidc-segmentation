
## Quick Links

- [Paper](https://drive.google.com/file/d/1bTLZDiEBEa07ajSySLb1_KffAw6OFiU_/view?usp=sharing)
- [Poster](https://drive.google.com/file/d/10KH6hrYE4J8k_6ep9odKOyde56Urh13k/view?usp=sharing)
- [Code](https://github.com/kzhang-20/lidc-segmentation)

Our research paper is written in IEEE-CPVR style. Our poster is in standard 32x24 format and was presented to our professors and TAs. We received full marks on this project for UW CSE's graduate-level [deep learning course](https://courses.cs.washington.edu/courses/cse493g1/23sp/).

## Research Highlights

### Motivation

Lung cancer has a mere 18% survival rate in the United States, largely due to the lack of preventive screening. The current method for screening lung cancer relies on radiologists, who are already in [short supply](https://www.acr.org/Practice-Management-Quality-Informatics/ACR-Bulletin/Articles/March-2022/The-Radiology-Labor-Shortage), to manually segment nodules on CT scans. Computer-aided diagnosis (CAD) systems can help alleviate the shortage of screening capacity, but are still lacking in accuracy.

> “the global implementation of lung cancer screening is of utmost importance” ~ International Association for the Study of Lung Cancer ([IASLC](https://doi.org/10.1016/j.jtho.2021.11.008)).

Given the extremely small size of biomedical datasets (LIDC only includes 1018 CT scans), data augmentations intuitively seem like a potent tool. Existing research confirms this intuition, but points out that each biomedical task has differnet optimal augmentations. Crucially, there has been **no existing work** in finding the best augmentations for lung nodule segmentation. Our paper explores this open question and provides recommendations for future scientists working with the LIDC dataset.

### Pipeline

We introduced the first open-source "plug-and-play" pipeline for the LIDC dataset, entirely in PyTorch. This contribution enables the research community to easily try out different approaches to the LIDC dataset (whether it be expanding the neural network architecure, utilizing different augmentations, training on with a new loss function,  or really anything supported by PyTorch) *without* having to deal with the confusing world that is medical imaging. 

> In essence, our pipeline abstracts away the medical part of problem task and lets researchers treat it as a normal deep learning task.

Under the hood, our workflow follows this training procedure:
 
1. Download the 125 GB [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).
2. Utilize [pylidc](https://pylidc.github.io) and [Pidicom](https://github.com/pydicom/pydicom) to extract 10,005 2D slices of 512 x 512 pixel arrays and convert them to PNG images.
3. For the detection task, store each nodule's bounding box cooordinates in [MS-COCO format](https://cocodataset.org/#format-data).
4. For the segmentation task, crop each nodule image and mask with a 64 x 64 pixel bounding box.
5. Extract regions of interest (ROIs) using either [Faster R-CNN](https://arxiv.org/abs/1506.01497) or [YOLO](https://arxiv.org/abs/1506.02640) (our pipeline supports both).
6. Augment images using any of [PyTorch's native augmentations](https://pytorch.org/vision/main/transforms.html). We also added support for [RandAugment](https://arxiv.org/abs/1909.13719), [AutoAugment](https://arxiv.org/abs/1805.09501), and [TrivialAugment](https://arxiv.org/abs/2103.10158).
7. Utilize our custom-designed U-Net or another PyTorch-compatiable architecture for the LIDC task.

### Models

### Augmentations

### Conclusion

Contrary to the suggestions of existing literature, augmentations provide only minor benefits for lung nodule segmentation. Future models should simply use horizontal and vertical flipping for augmentations and instead focus their efforts on improving the underlying network architecture.

We also bleieve that the development of a comprehensive automatic augmentation suite tailored to medical images would be of great utility. Similarly, efforts to increase the training data set by using GANs should also be explored further.

## Contributors

- [Tanush Yadav](https://www.linkedin.com/in/tanushyadav/)
- [Kevin Zhang](mailto:kzhang20@cs.washington.edu)
- [Marius Rakickas](https://www.linkedin.com/in/marius-rakickas/)

## Acknowledgements

This work would not have been possible without the guidance of the CSE 493G1 teaching team and the Allen Center `prost` and `senna` undergraduate GPUs.
