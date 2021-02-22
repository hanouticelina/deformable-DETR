


# Deformable DETR: Deformable Transformers for End-to-End Object Detection


This repository contains an implementation of the paper "Deformable Transformers for End-to-End Object Detection" : https://arxiv.org/pdf/2010.04159.pdf 


## Requirements

`pip install -r requirements.txt`

## Dataset
The `coco_extraction.py` script provides functions for creating an annotation file for coco with only the specified class indexes. It removes all images and bounding boxes not containning at least one of those classes. We kept only five randomly sampled classes from 91 available, `bear`, `bus`, `tie`, `toilet` and `vase`. The annotation files are `datasets/coco_light/coco_light_train.json` for the train set and `datasets/coco_light/coco_light_train.json` for the validation set. `Coco lignt` contains ~ 15K images in the train set and 656 images on the validation set.


## Usage 

`git clone https://github.com/hanouticelina/deformable-DETR.git`

`cd deformable-DETR`

### Training

The command for training Deformable DETR is as following:

`python main.py --enc_layers 3 --dec_layers 3 --batch_size 1`

Training convergence takes 72 GPU hours on a single GPU GeForce RTX 2080.

### Evaluation

To evaluate Deformable DETR on a subset of COCO 2017 validation set with a single GPU run:

`<path to config file> --resume <path to pre-trained model> --eval`


### Additional information

We provide scratch implementation of the following modules : 

* `deformable_transformer.py`, `decoder.py`, `encoder.py`, `deformable_detr.py` and `MultiHeadAttention.py`.

the remaining modules are mainly copied from the original DETR implementation : https://github.com/facebookresearch/detr

Pre-trained model can be found at : https://www.dropbox.com/s/vnkbfrui1ldwtah/checkpoint.pth?dl=0


## References
Xizhou Zhu et al., Deformable Transformers for End-to-End Object Detection.

Nicolas Carion et al., End-to-End Object Detection with Transformers.
