#!/bin/bash

# Set the new values for the variables
new_dataset='cora'
new_model='VGAE'
new_input_dim=1000
new_hidden1_dim=64
new_hidden2_dim=32
new_hidden3_dim=16
new_num_class=10
new_use_feature=false
new_num_epoch=300
new_learning_rate=0.001

# Replace the values in the args.py file
sed -i "s/^dataset = .*/dataset = '$new_dataset'/" args.py
sed -i "s/^model = .*/model = '$new_model'/" args.py
sed -i "s/^input_dim = .*/input_dim = $new_input_dim/" args.py
sed -i "s/^hidden1_dim = .*/hidden1_dim = $new_hidden1_dim/" args.py
sed -i "s/^hidden2_dim = .*/hidden2_dim = $new_hidden2_dim/" args.py
sed -i "s/^hidden3_dim = .*/hidden3_dim = $new_hidden3_dim/" args.py
sed -i "s/^num_class = .*/num_class = $new_num_class/" args.py
sed -i "s/^use_feature = .*/use_feature = $new_use_feature/" args.py
sed -i "s/^num_epoch = .*/num_epoch = $new_num_epoch/" args.py
sed -i "s/^learning_rate = .*/learning_rate = $new_learning_rate/" args.py

echo "Values in args.py file replaced with new values."


python3 train.py


# input_dim = 500
# hidden1_dim = 32
# hidden2_dim = 16
# num_class = 3

