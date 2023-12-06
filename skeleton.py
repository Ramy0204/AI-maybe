from numpy import *

class Network:
    def __init__(self, layer_sizes : list):
        
        self.ls = layer_sizes
        self.weights = [random.randn(self.ls[i], self.ls[i-1]) for i in range(1, len(self.ls))]
        self.biases = [random.randn(self.ls[i], 1) for i in range(1, len(self.ls))]
    def forward(self, input : list):
        a = array(input).reshape(len(input), 1)
        z = add((self.weights[0] @ a),self.biases[0])
        a = sigma(z)
        for i in range(1, len(self.biases)):
            z = add(self.weights[i]@a, self.biases[i])
            a = sigma(z)
        return a
    def backprop(self, x : list, y : list, learning_rate):
        x = array(x).reshape(len(x), 1)
        y = array(y).reshape(len(y), 1)
        z = add(self.weights[0]@(x),self.biases[0])
        a_s = [x]
        zs = [z]
        a = sigma(z)
        a_s.append(a)
        for i in range(1, len(self.ls)-1):
            z = add(self.weights[i]@a, self.biases[i])
            zs.append(z)
            a = sigma(z)
            a_s.append(a)
        delta_L = multiply((y-a_s[-1]),sigma_prime(zs[-1]))
        deltas = [delta_L]
        delta_l = delta_L
        for i in range(1, len(self.weights)):
            delta_l = multiply(((self.weights[-i]).transpose())@deltas[-1],sigma_prime(zs[-i-1]))
            deltas.append(delta_l)
        Nabla_w = [multiply(a_s[1],deltas[-1])]
        Nabla_n = deltas
        for i in range(2, len(a_s)):
            Nabla_w += [multiply(a_s[i],deltas[-i])]
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate*Nabla_w[i]
            self.biases[i] = self.biases[i] - learning_rate*Nabla_n[-i-1]
    def training(self, training_data : tuple, epochs, learning_rate):
        print("Training started")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} / {epochs}")
            for data in training_data:
                (x, y) = data
                self.backprop(x,y,learning_rate)
        print("Training Ended")
    def evaluate_accuracy(self, testing_data):
        cost = 0
        i = 0
        for x, y in testing_data:
            a = self.forward(x)
            cost += (linalg.norm(y-a))
            i += 1
        return ((1/(i))*cost)



        

def sigma(x):
    return 1.0 / (1.0 + exp(-x))
    

def sigma_prime(x):
    return sigma(x)*(sigma(x)-1) 

# a = Network([1,2,1])
# print(a.forward([0.9]))
# a.training([([0.9],[0.2])], 1000)
# print(a.forward([0.9]))

