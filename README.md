# DA6401
This repository is for the assignments of DA6401(Introduction to Deep Learning) 
### Task
The goal is to implement a feed forward network with the use of backpropagation from scratch on the fashion-MNIST dataset

### Submission 
My project --https://wandb.ai/ma23c047-indian-institute-of-technology-madras/DA6401%20-%20Assignment%201

Report --- https://wandb.ai/ma23c047-indian-institute-of-technology-madras/DA6401%20-%20Assignment%201/reports/Copy-of-ma23c047-s-DA6401-Assignment-1--VmlldzoxMTY3MDkwNQ

### Dataset
Two datasets have been used in the assignment Fashion MNIST and MNIST


from keras.datasets import fashion_mnist

from keras.datasets import mnist


### Supported Optimizers
It includes several advanced optimizers for efficient training:

Stochastic Gradient Descent (SGD)

Momentum SGD

Nesterov Accelerated Gradient (NAG)

RMSProp

Adam (Adaptive Moment Estimation)

Nadam (Nesterov Adaptive Moment Estimation)

### Loss Functions (Criteria)
Available loss functions include:

Cross Entropy Loss

Mean Squared Error (MSE)

### Backpropagation Methodology
Backpropagation is implemented iteratively, computing errors (delta) per layer and calculating gradients by multiplying deltas with inputs.

### Customization and Flexibility
The implementation allows extensive customization options:

Layer-specific activation functions

Sparse connectivity between layers

Variable neuron counts per layer

Adjustable input batch sizes

Customizable output activation functions

This flexibility enables users to experiment extensively with different neural network configurations and optimization strategies.

### Tools & Libraries Used
Tool/Library	Version	Purpose

Python	3.11.9	

NumPy	2.2.3	

Matplotlib	3.10.1	

Keras	3.9.0	Dataset loading and preprocessing

Weights & Biases (WandB)	0.19.8 Experiment tracking, visualization, hyperparameter tuning

Scikit-learn for	Data splitting, evaluation metrics, confusion matrices


### Usage Guide
To use the neural network implementation:

Define your network configuration in neural_network.py.

Select desired optimizer and loss function.

Run training scripts provided in the repository.

Track experiments using WandB dashboard for enhanced visualization.

