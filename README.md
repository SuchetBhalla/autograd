# MyTorch
An autograd engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The goal of this project is to understand the ML framework [PyTorch](https://github.com/pytorch/pytorch), by rebuilding it from scratch.

Eventually, this project will utilize CUDA for accelerated deep learning.

## Features
- _Value_ objects, which track gradients
- Operator overloading (+, -, *, /, **)
- Backpropagation through computational graphs
- Basic loss functions: MSE, BCE
- Parameter optimization with: Batch Gradient Descent

## Roadmap
- Support for the optimizer [ADAM](https://arxiv.org/abs/1412.6980)
- GPU support with CUDA

## Example usage
run _python -m test.test_
