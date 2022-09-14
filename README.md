![Python Depend](https://img.shields.io/badge/Python-3.6-blue) ![TF Depend](https://img.shields.io/badge/TensorFlow-2.6-orange)

# UVNQ_tf2
Implementation of **Uniform Variational Network Quantizer (UVNQ)** in **'[Quantization-Aware Pruning Criterion for Industrial Applications](https://ieeexplore.ieee.org/document/9398534)'**,<br>
which is publicated in IEEE Transactions on Industrial Electronics in March 2022. 

# How to use
1. First pretrain network (UVNQLeNet_5 or UVNQMLP) (set pretrain = True in train_mnist.py)
2. Load the pretrained weights and sparsify the network (set pretrain = False)
3. N is number of bits to quantize the network (N>=1, N = 4 is the best)  
4. Beta is the UVNQ hyperparameter (beta = 1.5)


# Warning
This is **NOT** the UVNQ code that used in the experiments in the paper. There's some change in this code from the code used in the paper.<br>
This code can also quantize the network yet you may not be able to reproduce the experiment results in the paper. 
