import numpy as np

class Neurallayer:

    def __init__(self, n_input, n_neurons, activation, initializer="xavier"):
        self.activation = activation
        if initializer == "random":
            self.weights = np.random.randn(n_input, n_neurons)
        elif initializer == "xavier":
            self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2.0/(n_input + n_neurons))
        self.bias = np.zeros(n_neurons)

        self.h_weights = np.zeros_like(self.weights)
        self.h_bias = np.zeros(n_neurons)
        self.m_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros(n_neurons)

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        return self.activation.activation(self.z)
    
    def backward(self, w, delta):
        if w is None:
            self.delta = delta
        else:
            error = np.matmul(delta, w.T)
            self.delta = error * self.activation.derivative(self.z)
        self.dw = np.matmul(self.x.T, self.delta)
        self.db = np.sum(self.delta, axis=0)
        
class Optimizer:

    def __init__(self, learning_rate, method, config):
        self.method = method
        self.learning_rate = learning_rate
        self.config = config
    
    def update(self, layer):
        if self.method == "sgd":
            return self.sgd(layer)
        
        elif self.method == "momentum":
            return self.momentum(layer)
        
        elif self.method == "nag":
            return self.nag(layer)
        
        elif self.method == "rmsprop":
            return self.rmsprop(layer)
        
        elif self.method == "adam":
            return self.adam(layer)
        
        elif self.method == "nadam":
            return self.nadam(layer)

    def sgd(self, layer):
        layer.weights = layer.weights - self.learning_rate * layer.dw
        layer.bias = layer.bias - self.learning_rate * layer.db
    
    def momentum(self, layer):
        layer.h_weights = self.config["momentum"] * layer.h_weights + layer.dw
        layer.h_bias = self.config["momentum"] * layer.h_bias + layer.db
        layer.weights -= self.learning_rate * (layer.h_weights + self.config["decay"] * layer.weights)
        layer.bias -= self.learning_rate * (layer.h_bias + self.config["decay"]*layer.bias)
    
    def nag(self, layer):
        layer.h_weights = self.config["momentum"] * layer.h_weights + layer.dw
        layer.h_bias = self.config["momentum"] * layer.h_bias + layer.db
        layer.weights -= self.learning_rate * (self.config["momentum"] * layer.h_weights + layer.dw + self.config["decay"] * layer.weights)
        layer.bias -= self.learning_rate * (self.config["momentum"] * layer.h_bias + layer.db + self.config["decay"] * layer.bias)

    def rmsprop(self, layer):
        layer.h_weights = self.config["beta"] * layer.h_weights + (1 - self.config["beta"]) * layer.dw**2
        layer.h_bias = self.config["beta"] * layer.h_bias + (1 - self.config["beta"]) * layer.db**2
        layer.weights -= self.learning_rate * (layer.dw / (np.sqrt(layer.h_weights) + self.config["epsilon"])) + self.config["decay"] * layer.weights * self.learning_rate
        layer.bias -= self.learning_rate * (layer.db / (np.sqrt(layer.h_bias) + self.config["epsilon"])) + self.config["decay"] * layer.bias * self.learning_rate

    def adam(self, layer):
        layer.m_weights = self.config["beta1"] * layer.m_weights + (1 - self.config["beta1"]) * layer.dw
        layer.m_bias = self.config["beta1"] * layer.m_bias + (1 - self.config["beta1"]) * layer.db
        layer.h_weights = self.config["beta2"] * layer.h_weights + (1 - self.config["beta2"]) * layer.dw**2
        layer.h_bias = self.config["beta2"] * layer.h_bias + (1 - self.config["beta2"]) * layer.db**2
        corterm1 = 1/(1 - self.config["beta1"]**(self.config["timestep"] + 1))
        corterm2 = 1/(1 - self.config["beta2"]**(self.config["timestep"] + 1))
        w_hat1 = layer.m_weights * corterm1 
        b_hat1 = layer.m_bias * corterm1
        w_hat2 = layer.h_weights * corterm2
        b_hat2 = layer.h_bias * corterm2
        layer.weights -= self.learning_rate * (w_hat1 / ((np.sqrt(w_hat2)) + self.config["epsilon"])) + self.config["decay"] * layer.weights * self.learning_rate
        layer.bias -= self.learning_rate * (b_hat1 / ((np.sqrt(b_hat2)) + self.config["epsilon"])) + self.config["decay"] * layer.bias * self.learning_rate   

    def nadam(self, layer):
        layer.m_weights = self.config["beta1"] * layer.m_weights + (1 - self.config["beta1"]) * layer.dw
        layer.m_bias = self.config["beta1"] * layer.m_bias + (1 - self.config["beta1"]) * layer.db
        layer.h_weights = self.config["beta2"] * layer.h_weights + (1 - self.config["beta2"]) * layer.dw**2
        layer.h_bias = self.config["beta2"] * layer.h_bias + (1 - self.config["beta2"]) * layer.db**2
        corterm1 = 1/(1 - self.config["beta1"]**(self.config["timestep"] + 1))
        corterm2 = 1/(1 - self.config["beta2"]**(self.config["timestep"] + 1))
        w_hat1 = layer.m_weights * corterm1 
        b_hat1 = layer.m_bias * corterm1
        w_hat2 = layer.h_weights * corterm2
        b_hat2 = layer.h_bias * corterm2
        combined_weight_update = self.config["beta1"] * w_hat1 + ((1 - self.config["beta1"]) / (1 - self.config["beta1"] ** (self.config["timestep"] + 1))) * layer.dw
        combined_bias_update = self.config["beta1"] * b_hat1 + ((1 - self.config["beta1"]) / (1 - self.config["beta1"] ** (self.config["timestep"] + 1))) * layer.db
        layer.weights -= self.learning_rate * (combined_weight_update / ((np.sqrt(w_hat2)) + self.config["epsilon"])) + self.config["decay"] * layer.weights * self.learning_rate
        layer.bias -= self.learning_rate * (combined_bias_update / ((np.sqrt(b_hat2)) + self.config["epsilon"])) + self.config["decay"] * layer.bias * self.learning_rate 
class Loss:

    def __init__(self,method):
        self.method = method
    
    def loss(self, y_true, y_pred):
        if self.method == "mse":
            return self.mse(y_true, y_pred)
        elif self.method == "bce":  
            return self.bce(y_true, y_pred)
        elif self.method == "crossentropy":
            return self.crossentropy(y_true, y_pred)

    def derivative(self, y_true, y_pred):
        return  y_pred - y_true
    
    def mse(self, y_true, y_pred):
        return 0.5*np.mean(np.power(y_pred - y_true, 2))
    
    def bce(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def crossentropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred))
    
class Activation:

    def __init__(self, method):
        self.method = method
    
    def activation(self, x):
        if self.method == "sigmoid":
            return self.sigmoid(x)
        elif self.method == "relu":
            return self.relu(x)
        elif self.method == "softmax": 
            return self.softmax(x)
        elif self.method == "tanh":
            return self.tanh(x)
                
    def derivative(self, x):
        if self.method == "sigmoid":
            return self.sigmoid_derivative(x)
        elif self.method == "relu":
            return self.relu_derivative(x)
        elif self.method == "softmax": 
            return self.softmax_derivative(x)   
        elif self.method == "tanh":
            return self.tanh_derivative(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    def softmax_derivative(self, x):
        return np.ones_like(x)
    
    def tanh_derivative(self, x):
        return 1 - np.power(self.tanh(x), 2)
    
class NeuralNetwork:
    
    def __init__(self, n_input, n_output, n_hidden, n_layers, activation, initializer="xavier"):
        self.layer = []
        # Create output activation for last layer
        self.output_activation = Activation("softmax")
        
        for i in range(n_layers):
            if i == 0:
                self.layer.append(Neurallayer(n_input, n_hidden, activation, initializer=initializer))
            elif i == n_layers - 1:
                # Use softmax activation for output layer
                self.layer.append(Neurallayer(n_hidden, n_output, self.output_activation, initializer=initializer))
            else:
                self.layer.append(Neurallayer(n_hidden, n_hidden, activation, initializer=initializer))
        
    def feedforward(self, x):
        for layer in self.layer:
            x = layer.forward(x)
        return x
    
    def backpropagation(self, y, y_hat, loss_derivative, optimizer):
        # Gradient Computation
        for i in reversed(range(len(self.layer))):
            if i == len(self.layer) - 1:
                self.layer[i].backward(None, loss_derivative(y, y_hat))
            else:
                self.layer[i].backward(self.layer[i + 1].weights, self.layer[i + 1].delta)
        # Weight Update
        for i in reversed(range(len(self.layer))):    
            optimizer.update(self.layer[i])

seed = 42
np.random.seed(seed)


import matplotlib.pyplot as plt


def load_fashion_mnist():
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot

def calculate_accuracy(y_true, y_pred):
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == true_classes)

def train_fashion_mnist(nn, optimizer, criterion, epochs=10, batch_size=32):
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    train_loss_hist = []
    train_accuracy_hist = []
    test_loss_hist = []
    test_accuracy_hist = []
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        epoch_losses = []
        
        # Batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Forward pass
            y_pred = nn.feedforward(X_batch)
            loss = criterion.loss(y_batch, y_pred)
            epoch_losses.append(loss)
            
            # Backward pass
            nn.backpropagation(y_batch, y_pred, criterion.derivative, optimizer)
            optimizer.config["timestep"] += 1
        
        # Calculate training metrics
        train_preds = nn.feedforward(X_train)
        train_loss = criterion.loss(y_train, train_preds)
        train_accuracy = calculate_accuracy(y_train, train_preds)
        
        # Calculate test metrics
        test_preds = nn.feedforward(X_test)
        test_loss = criterion.loss(y_test, test_preds)
        test_accuracy = calculate_accuracy(y_test, test_preds)
        
        # Store metrics
        train_loss_hist.append(train_loss)
        train_accuracy_hist.append(train_accuracy)
        test_loss_hist.append(test_loss)
        test_accuracy_hist.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n')
    
    return train_loss_hist, train_accuracy_hist, test_loss_hist, test_accuracy_hist


activation = Activation("sigmoid")
optimizer = Optimizer(learning_rate=0.01, method="sgd", config={
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "timestep": 0,
    "decay": 0.0
})
criterion = Loss("crossentropy")
nn = NeuralNetwork(n_input=784, n_output=10, n_hidden=128, n_layers=3, activation=activation)
histories = train_fashion_mnist(nn, optimizer, criterion, epochs=15, batch_size=32)

def plot_training_history(histories):
    train_loss_hist, train_accuracy_hist, test_loss_hist, test_accuracy_hist = histories
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(test_loss_hist, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_hist, label='Train Accuracy')
    plt.plot(test_accuracy_hist, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(histories)