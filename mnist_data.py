import mnist_loader
import skeleton
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = skeleton.Network([784, 30, 10])
net.training(training_data, 10000, 1000)
print(net.evaluate_accuracy(test_data))