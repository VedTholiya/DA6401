import argparse
import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
import wandb
from nnl import NeuralNetwork, Optimizer, Loss, Activation

# Default configuration
DEFAULT_CONFIG = {
    'wandb_project': "DA6401 - Assignment 1",
    'wandb_entity': "ma23c047-indian-institute-of-technology-madras",
    'dataset': 'fashion_mnist',
    'num_layers': 1,
    'hidden_size': 4,
    'activation': 'relu',
    'learning_rate': 0.1,
    'optimizer': 'adam',
    'batch_size': 4,
    'epochs': 1,
    'loss': 'cel',
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'weight_decay': 0.0,
    'weight_init': 'xavier',
    'momentum': 0.5,
    'beta': 0.5,
    'n_input': 784,
    'n_output': 10
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default=DEFAULT_CONFIG['wandb_project'])
    parser.add_argument("-we", "--wandb_entity", type=str, default=DEFAULT_CONFIG['wandb_entity'])
    parser.add_argument("-d", "--dataset", type=str, choices=['mnist', 'fashion_mnist'], 
                       default='fashion_mnist')
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", type=str, 
                       choices=['mse', 'bce', 'crossentropy'],  
                       default='crossentropy')
    parser.add_argument("-o", "--optimizer", type=str,
                       choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                       default='sgd')
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", type=str,
                       choices=['random', 'Xavier'], default='random')
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str,
                       choices=['sigmoid', 'relu', 'tanh', 'softmax'], 
                       default='sigmoid')
    
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    
    # Map CLI arguments to config
    config.update({
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'dataset': args.dataset,
        'n_layers': args.num_layers,
        'n_hidden': args.hidden_size,
        'activation': args.activation.lower(),
        'learning_rate': args.learning_rate,
        'optimizer_method': args.optimizer,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'criterion': args.loss,  
        'beta1': args.beta1,
        'beta2': args.beta2,
        'epsilon': args.epsilon,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'beta': args.beta,
        'initializer': 'xavier' if args.weight_init.lower() == 'xavier' else 'random'  # Match nnl.py initializer
    })
    
    return config

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
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

def train(config):
    # Initialize wandb
    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)
    
    # Load data
    X_train, y_train, X_test, y_test = get_dataset(config['dataset'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Initialize model components with correct parameters
    activation = Activation(config['activation'])
    criterion = Loss(config['criterion'])
    optimizer = Optimizer(
        learning_rate=config['learning_rate'],
        method=config['optimizer_method'],
        config={
            "beta1": config['beta1'],
            "beta2": config['beta2'],
            "epsilon": config['epsilon'],
            "timestep": 0,
            "decay": config['weight_decay'],  
            "momentum": config['momentum'],
            "beta": config['beta']
        }  
    )  
    
    nn = NeuralNetwork(
        n_input=config['n_input'],
        n_output=config['n_output'],
        n_hidden=config['n_hidden'],
        n_layers=config['n_layers'],
        activation=activation,
        initializer=config['initializer']  
    )  
    
    # Training loop
    for epoch in range(config['epochs']):
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in range(0, len(X_train), config['batch_size']):
            X_batch = X_train[i:i+config['batch_size']]
            y_batch = y_train[i:i+config['batch_size']]
            
            y_pred = nn.feedforward(X_batch)
            nn.backpropagation(y_batch, y_pred, criterion.derivative, optimizer)
            optimizer.config["timestep"] += 1
        
        # Calculate metrics
        train_preds = nn.feedforward(X_train)
        train_loss = criterion.loss(y_train, train_preds)
        train_accuracy = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1))
        
        val_preds = nn.feedforward(X_val)
        val_loss = criterion.loss(y_val, val_preds)
        val_accuracy = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))
        
        # metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")
    
    # Final test evaluation
    test_preds = nn.feedforward(X_test)
    test_loss = criterion.loss(y_test, test_preds)
    test_accuracy = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
    
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    })
    
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return nn

if __name__ == "__main__":
    config = parse_arguments()
    print("\nTraining with configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("\n")
    
    model = train(config)
