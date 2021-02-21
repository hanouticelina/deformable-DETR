


# Deformable DETR: Deformable Transformers for End-to-End Object Detection


This repository contains an implementation of the paper "Deformable Transformers for End-to-End Object Detection" : https://arxiv.org/pdf/2010.04159.pdf 


## Requirements

`pip install -r requirements.txt`

## Dataset



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

## Results

We compared our model to state of the art object detectors pre-trained on COCO 2017 train set and fine-tuned on the subset on which our model was trained. The results are depicted on the table below. 


## References
Xizhou Zhu et al., Deformable Transformers for End-to-End Object Detection.

Nicolas Carion et al., End-to-End Object Detection with Transformers.
