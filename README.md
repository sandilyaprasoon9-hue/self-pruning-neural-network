# Self-Pruning Neural Network (CIFAR-10)

This project implements a neural network that learns to prune its own weights during training using learnable gate parameters and L1 regularization.

## Overview

In many real-world applications, large neural networks are expensive in terms of memory and computation. This project explores a self-pruning mechanism where the model automatically removes less important connections during training.

Each weight in the network is associated with a learnable gate value, which controls whether the weight is active or suppressed.

## Key Idea

Each weight is modified as:

pruned_weights = weight * sigmoid(gate_scores)

- gate_scores are learnable parameters  
- sigmoid ensures values between 0 and 1  
- values close to 0 effectively remove the weight  

## Sparsity Mechanism

The loss function is defined as:

Total Loss = CrossEntropyLoss + λ * SparsityLoss

- SparsityLoss is the sum of all gate values  
- L1 regularization encourages gate values to move toward zero  
- When a gate approaches zero, the corresponding weight is effectively pruned  

## Results

| Lambda | Test Accuracy | Sparsity (%) |
 
| 1e-7   | 50.30%     | 16.67%      |
| 1e-6   | 51.91%     | 71.93%      |
| 1e-5   | 53.92%     | 75.11%      |

## Observations

- Increasing λ increases sparsity significantly  
- Gate values shift toward zero during training  
- λ = 1e-6 provides a good balance between accuracy and sparsity  
- The model successfully removes a large number of unnecessary connections  

## Gate Distribution

The distribution of gate values shows a large number of values close to zero, indicating that many weights are pruned. At the same time, some values remain higher, meaning important connections are preserved.

## Dataset

- CIFAR-10  
- Loaded using torchvision.datasets  

## Model Architecture

- Fully connected neural network  
- Custom PrunableLinear layers  
- ReLU activations  

## Tech Stack

- Python  
- PyTorch  
- Matplotlib  

## Repository Structure

self-pruning-neural-network/
│
├── self_pruning_net.ipynb
├── README.md

## Conclusion

This project demonstrates that a neural network can learn to prune itself during training using L1 regularization on gate values. It achieves high sparsity while maintaining reasonable accuracy, showing a clear trade-off between model efficiency and performance.

## Author

Prasoon Jha
