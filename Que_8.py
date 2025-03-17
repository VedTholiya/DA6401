import wandb
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from nnl import NeuralNetwork, Optimizer, Loss, Activation

sweep_config = {
    'method': 'random',
    'name': 'NN_SWEEP_CEL',  # Modified name
    'metric': {
        'name': 'cel_val_accuracy',  # Modified metric name
        'goal': 'maximize'
    },
    'parameters': {
        'n_layers': {
            'values': [2, 3, 4]
        },
        'n_hidden': {
            'values': [64, 128, 256]
        },
        'activation': {
            'values': ['sigmoid', 'relu', 'tanh']
        },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'optimizer_method': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'epochs': {
            'values': [5, 10, 15]
        },
        'beta1': {
            'value': 0.9
        },
        'beta2': {
            'value': 0.999
        },
        'epsilon': {
            'value': 1e-8
        },
        'decay': {
            'values': [0.0, 0.5, 0.0005]
        },
        'criterion': {
            'values': ['cross_entropy']  # Changed to cross-entropy
        }
    }
}

def wandb_sweep():
    run = wandb.init()
    config = wandb.config
    
    run.name = f"cel_l_{config.n_layers}_h_{config.n_hidden}_act_{config.activation}_lr_{config.learning_rate}_bs_{config.batch_size}_opt_{config.optimizer_method}_d_{config.decay}"
    
    activation = Activation(config.activation)
    criterion = Loss(config.criterion)
    optimizer = Optimizer(
        learning_rate=config.learning_rate,
        method=config.optimizer_method,
        config={
            "beta1": config.beta1,
            "beta2": config.beta2,
            "epsilon": config.epsilon,
            "timestep": 0,
            "decay": config.decay,
            "momentum": 0.9,
            "beta": 0.9
        }
    )
    
    nn = NeuralNetwork(
        n_input=784,
        n_output=10,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        activation=activation
    )
    
    histories = train_fashion_mnist(
        nn=nn,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.epochs,
        batch_size=config.batch_size,
        wandb_logging=True
    )
    
    return histories

# Load fashion MNIST and calculate accuracy functions remain the same
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot

def calculate_accuracy(y_true, y_pred):
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == true_classes)

def train_fashion_mnist(nn, optimizer, criterion, epochs=10, batch_size=32, wandb_logging=False):
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            y_pred = nn.feedforward(X_batch)
            nn.backpropagation(y_batch, y_pred, criterion.derivative, optimizer)
            optimizer.config["timestep"] += 1
        
        train_preds = nn.feedforward(X_train)
        train_loss = criterion.loss(y_train, train_preds)
        train_accuracy = calculate_accuracy(y_train, train_preds)
        
        val_preds = nn.feedforward(X_val)
        val_loss = criterion.loss(y_val, val_preds)
        val_accuracy = calculate_accuracy(y_val, val_preds)
        
        train_loss_hist.append(train_loss)
        train_accuracy_hist.append(train_accuracy)
        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_accuracy)
        
        if wandb_logging:
            wandb.log({
                "epoch": epoch + 1,
                "cel_train_loss": train_loss,
                "cel_train_accuracy": train_accuracy,
                "cel_val_loss": val_loss,
                "cel_val_accuracy": val_accuracy
            })
    
    test_preds = nn.feedforward(X_test)
    test_loss = criterion.loss(y_test, test_preds)
    test_accuracy = calculate_accuracy(y_test, test_preds)
    
    if wandb_logging:
        wandb.log({
            "cel_test_loss": test_loss,
            "cel_test_accuracy": test_accuracy
        })
    
    return train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist

if __name__ == "__main__":
    wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
    
    sweep_id = wandb.sweep(sweep_config, project="DA6401 - Assignment 1")
    
    wandb.agent(sweep_id, function=wandb_sweep, count=50)
    
    api = wandb.Api()
    sweep = api.sweep(f"ma23c047-indian-institute-of-technology-madras/DA6401 - Assignment 1/{sweep_id}")
    best_run = sweep.best_run()
    
    print("\nBest Configuration:")
    print("-------------------")
    print(f"Best Validation Accuracy: {best_run.summary.get('cel_val_accuracy'):.4f}")
    print(f"Parameters:")
    for key, value in best_run.config.items():
        print(f"{key}: {value}")
