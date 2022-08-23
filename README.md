# patcher

Pytorch Implementation for our MICCAI 2022 paper: [Patcher: Patch Transformers with Mixture of Experts for Precise Medical Image Segmentation
.](https://arxiv.org/abs/2206.01741)



# Overview
![Loading LambdaUnet Overview](https://github.com/YanglanOu/LambdaUNet/blob/main/images/lambda_layer.png)

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
We will release the dataset soon.

# Training
You can train your own models with your customized configs and dataset. For example:

```
python train.py --cfg config_file 
```

# Acknowledgment
This repo borrows code from
* [SETR](https://github.com/fudan-zvg/SETR)


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

