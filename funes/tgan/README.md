# timegan-pytorch
This repository holds the code for the reimplementation of TimeGAN ([Yoon et al., NIPS2019](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks)) using PyTorch. Some of the code was derived from the original implementation [here](https://github.com/jsyoon0823/TimeGAN).

> :warning: WARNING!!!
> - This implementation is written for other purposes, not for experiments in the original paper.
> - There are some known issues that I've haven't got time to resolve (see issue [#1](https://github.com/d9n13lt4n/timegan-pytorch/issues/1#issuecomment-895126605)).

## Getting Started
### Installing Requirements
This implementation assumes Python3.8 and a Linux environment with a GPU is used.
```bash
cat requirements.txt | xargs -n 1 pip install --upgrade
```

