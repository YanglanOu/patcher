# Patcher

Pytorch Implementation for our MICCAI 2022 paper: [Patcher: Patch Transformers with Mixture of Experts for Precise Medical Image Segmentation
.](https://arxiv.org/abs/2206.01741)



# Overview
![Loading Patcher Overview](https://github.com/YanglanOu/patcher/blob/master/imgs/overview.png)

# Installation
### Environment
* Tested OS: Linux
* Python >= 3.6

### Dependencies:
1. Install [PyTorch 1.4.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt

    ```

### Datasets
Kvasir-SEG is available [online](https://datasets.simula.no/kvasir-seg/). Download, put it under data/ and run:
```
python pre_process.py 
```

We will also release the stroke lesion segmentation data soon. 
 

# Training
You can try our models on Kvasir-SEG. For example:

```
python train.py --cfg patchformer_kvasir_moe_2_strct_adam_aug2 --gpu 0,1
```

Or you can train our models on your customized data, just put them under data/ and follow the same scheme. 


# Acknowledgment
This repo borrows code from
* [SETR](https://github.com/fudan-zvg/SETR)
* [Another implementation of SETR](https://github.com/gupta-abhay/setr-pytorch)


# Citation
If you find our work useful in your research, please cite our paper:
```
@article{ou2022patcher,
  title={Patcher: Patch Transformers with Mixture of Experts for Precise Medical Image Segmentation},
  author={Ou, Yanglan and Yuan, Ye and Huang, Xiaolei and Wong, Stephen TC and Volpi, John and Wang, James Z and Wong, Kelvin},
  journal={arXiv preprint arXiv:2206.01741},
  year={2022}
}
```

