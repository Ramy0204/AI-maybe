# AI-maybe

This is a project I made in 2023 to learn the intricacies of neural networks, how they work and how to train them.

I wanted to understand what was actually happening inside a neural network instead of just using a framework that does everything for me.

## What I used to learn

I started with [3Blue1Brown's neural network videos](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), more specifically the first four videos of the playlist, to understand the mathematical part of deep learning.

Once I understood the mathematics well enough to explain it myself, I used Michael Nielsen's book [Neural Networks and Deep Learning](https://neuralnetworksanddeeplearning.com/) to start implementing what I had learned in Python.

I did not use any neural network frameworks for my implementation. I only used NumPy for the mathematical operations, as I wanted to write the network and its training process myself.

I also used Nielsen's MNIST dataset and data loader.

## What I implemented

The main part of the project is in `skeleton.py`, where I made a `Network` class that takes a list of layer sizes, allowing me to create networks with different numbers of layers and neurons.

I implemented the following:

- **Weights and biases:** Randomly initialized when creating the network.
- **Forward propagation:** Calculates the output of the network from an input using matrix multiplication and the sigmoid activation function.
- **Backpropagation:** My implementation of the process used to calculate adjustments to the weights and biases.
- **Training:** Goes through the training data over a specified number of epochs, updating the weights and biases.
- **Evaluation:** Tests the network on MNIST images and calculates the percentage of correctly recognized digits.

For example, in `mnist_data.py`, I used the following architecture:

**784 → 30 → 10**

The 784 neurons of the input layer represent the 784 pixels of a 28×28 MNIST image. The hidden layer contains 30 neurons, and the output layer contains 10 neurons, one for each digit from 0 to 9.

## The files

- `skeleton.py` — My implementation of the neural network, including forward propagation, backpropagation, training and evaluation.
- `mnist_loader.py` — The MNIST data loader from Michael Nielsen's learning resources.
- `mnist_data.py` — One of my experiments with training and evaluating the network on MNIST.
- `Stats.py` — An experiment to plot the accuracy of the network depending on the number of training epochs.
- `attempt.py` — Some additional functions I made to convert arrays into images and visualize the network's guesses.
- `data/` — Contains the MNIST dataset and some images.
- `models/` — Contains a saved network file.

## How to try it

The project is written in Python, with NumPy being the only library needed for the neural network itself. Some of the additional scripts also use Matplotlib and Pillow.

You can install these dependencies with:

```bash
pip install numpy matplotlib pillow
```

The main implementation is in `skeleton.py`. You can import the `Network` class, create a network by specifying its layer sizes, and use the training and evaluation methods with the MNIST data.

The other Python files are mostly experiments I made while working on the project, rather than a single finished application.

## What I learned

Making this project helped me understand how the mathematics I had learned actually translated into code.

In particular, I got a much better understanding of how the different layers interact, how the weights and biases affect the output, and how the network attempts to improve through training.

This was mainly a learning project, so the code is not necessarily the most efficient or the cleanest implementation. The point was to understand how everything worked by trying to make it myself.

## Credits

- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Michael Nielsen — Neural Networks and Deep Learning](https://neuralnetworksanddeeplearning.com/)
- [Michael Nielsen's original code repository](https://github.com/mnielsen/neural-networks-and-deep-learning)
