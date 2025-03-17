import wandb
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from nnl import NeuralNetwork, Optimizer, Loss, Activation

# Best configuration found from sweep
BEST_CONFIG = {
    'n_layers': 3,
    'n_hidden': 128,
    'activation': 'relu',
    'learning_rate': 0.001,
    'optimizer_method': 'adam',
    'batch_size': 128,
    'epochs': 15,
    'criterion': 'cel',
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'decay': 0.0
}

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot

def train_best_mnist():
    # Initialize wandb
    wandb.init(project="MNIST-Best-Config", name="mnist_best_validation", config=BEST_CONFIG)
    
    # Create model components
    activation = Activation(BEST_CONFIG['activation'])
    criterion = Loss(BEST_CONFIG['criterion'])
    optimizer = Optimizer(
        learning_rate=BEST_CONFIG['learning_rate'],
        method=BEST_CONFIG['optimizer_method'],
        config={
            "beta1": BEST_CONFIG['beta1'],
            "beta2": BEST_CONFIG['beta2'],
            "epsilon": BEST_CONFIG['epsilon'],
            "timestep": 0,
            "decay": BEST_CONFIG['decay'],
            "momentum": 0.9,
            "beta": 0.9
        }
    )
    
    nn = NeuralNetwork(
        n_input=784,
        n_output=10,
        n_hidden=BEST_CONFIG['n_hidden'],
        n_layers=BEST_CONFIG['n_layers'],
        activation=activation
    )
    
    # Load and split data
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Training loop
    for epoch in range(BEST_CONFIG['epochs']):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train, y_train = X_train[indices], y_train[indices]
        
        # Mini-batch training
        for i in range(0, len(X_train), BEST_CONFIG['batch_size']):
            X_batch = X_train[i:i+BEST_CONFIG['batch_size']]
            y_batch = y_train[i:i+BEST_CONFIG['batch_size']]
            
            y_pred = nn.feedforward(X_batch)
            nn.backpropagation(y_batch, y_pred, criterion.derivative, optimizer)
        
        # Calculate validation accuracy
        val_preds = nn.feedforward(X_val)
        val_accuracy = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))
        
        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "validation/accuracy": val_accuracy,
            "validation/step": epoch
        })
        
        print(f"Epoch {epoch+1}/{BEST_CONFIG['epochs']}, Validation Accuracy: {val_accuracy:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
    train_best_mnist()
