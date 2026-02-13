### Neural Network from Scratch using NumPy

Implemented a fully-connected neural network from scratch without using deep learning frameworks, and trained it to classify handwritten digits from the MNIST dataset.
The objective was to understand how neural networks work internally by implementing forward propagation, backpropagation, and gradient descent manually using only NumPy.

# Architecture
Input layer (784)
- Hidden layer 1 (ReLU)
- Hidden layer 2 (tanh)
- Output layer (Softmax)

# Implementation Details
- No PyTorch or TensorFlow used
- Forward pass and backpropagation implemented manually, from scratch, using only NumPy
- Cross-entropy loss
- Gradient descent optimization
- Random weight initialization

# Results
Achieved ~85% accuracy on the MNIST test set.

# How to Run
1. Install dependencies:
   pip install numpy matplotlib
2. Run:
   python devsoc_nn.py

# Key Learning
This project helped me deeply understand how backpropagation, gradients, and weight updates work internally instead of relying on high-level frameworks.
