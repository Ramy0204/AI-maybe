import matplotlib.pyplot as plt
import numpy as np
import mnist_loader
import skeleton
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


epochs = np.arange(0, 3000)
accuracy = np.zeros(epochs.size)

for i in epochs:
    net = skeleton.Network([784, 16, 16, 10])
    net.training(training_data, i, 0.1)
    accuracy[i] = net.evaluate_accuracy(test_data)

plt.plot(epochs, accuracy)