# DGRec + Changes

A PyTorch and DGL implementation for the WSDM 2023 paper below:  
[DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation](https://arxiv.org/pdf/2211.10486.pdf)

Environment
DGL version 1.0.1
Pytorch version 1.12.1

## Running
``python main.py``  
Then you can get similar result on TaoBao dataset as in the paper.  

You can check different hyper-parameters in `utils/parser.py`

## Changes

1) Adding transformations after and before each graph convolution
2) Change the Layer Attention to Gated Attention