# The Best of Both Worlds: On the Delimma of Out-of-Distribution Detection (NeurIPS'24)

This repo contains the code for our NeurIPS 2024 paper [The Best of Both Worlds: On the Delimma of Out-of-Distribution Detection](https://openreview.net/pdf?id=B9FPPdNmyk). We propose an learning objective for both OOD detection and generalization.

[TOC]

## Install Requirements

To get started with this repository, you'll need to follow these installation steps. Before proceeding, make sure you have anaconda installed.

```
pip3 install -r requirement.txt
```



## Data Preparation

We follow the [POEM](https://github.com/deeplearning-wisc/poem) and [OpenOOD](https://github.com/Jingkang50/OpenOOD/) to prepare the datasets. We provide links and instructions to download each dataset:

### ID datasets

- CIFAR10 and CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html

### Semantic OOD datasets

- [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`.
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
- [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365`.
- [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
- [LSUN-resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

These datasets are used to evaluate OOD detection performance.



## Training pipeline

### Step 1: Pretraining

Our DUL is devised in a finetune manner. Follow the below instructions for standard pretraining:

```
cd CIFAR
python baseline.py --dataset cifar10
```



## Step 2: Finetuning

Then, finetune the pretrain model with the following instructions for both OOD detection and generalization purposes:

```
python finetune.py --method DUL --dataset cifar10 --epoch 20
python finetune.py --method DUL --dataset cifar100 --epoch 30
```



## Evaluation

Evaluate the OOD detection and generalization performance with below instructions:

```
python test.py --dataset cifar10 --method DUL --score diff_entropy --epoch 19
python test.py --dataset cifar100 --method DUL --score diff_entropy --epoch 29
```



## TODO

We are actively working on releasing the code for ImageNet experiments.



## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Qingyang Zhang ([qingyangzhang@tju.edu.cn](qingyangzhang@tju.edu.cn)). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please cite our paper if you find the repo helpful in your work:

```
@inproceedings{zhang2024best,
   title={The Best of Both Worlds: On the Dilemma of Out-of-distribution Detection},
   author={Zhang, Qingyang and Feng, Qiuxuan and Zhou, Joey Tianyi and Bian, Yatao and Hu, Qinghua and Zhang, Changqing},
   booktitle={Advances in Neural Information Processing Systems},
   year={2024}
}
```



## Acknowledgement

The code is developed based on [Outlier Exposure](https://github.com/hendrycks/outlier-exposure),  [POEM](https://github.com/deeplearning-wisc/poem) and [OpenOOD](https://github.com/Jingkang50/OpenOOD/).