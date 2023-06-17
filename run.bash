#!/bin/bash

# VAE, VGAE, VGAE2, VGAE3, VGAE4, VGAE5, VGAE6, VGAE7, VGAE8, VGAE9, VDGAE
python3 train.py --dataset cora --model GAE --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE2 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE3 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE4 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE5 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE6 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE7 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE8 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE9 --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VDGAE --input_dim 1433 --num_class 7
python3 train.py --dataset cora --model VGAE4v2 --input_dim 1433 --num_class 7


# input_dim = 500
# hidden1_dim = 32
# hidden2_dim = 16
# num_class = 3

