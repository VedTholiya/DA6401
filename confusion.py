import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nnl import NeuralNetwork, Optimizer, Loss, Activation
from wandb_sweep import load_fashion_mnist

# Best configuration parameters
BEST_CONFIG = {
    'n_layers': 3,
    'n_hidden': 128,
    'activation': 'relu',
    'learning_rate': 0.001,
    'optimizer_method': 'nadam',
    'batch_size': 32,
    'epochs': 10,
    'criterion': 'mse',
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'decay': 0.0
}

def plot_confusion_matrix(y_true, y_pred):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Fashion MNIST Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()  

def main():
    # Initialize wandb
    wandb.init(project="DA6401 - Assignment 1", name="best_model_final", config=BEST_CONFIG)
    
    
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
    
    # Load data
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    # Training loop
    n_samples = X_train.shape[0]
    for epoch in range(BEST_CONFIG['epochs']):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Mini-batch training
        for i in range(0, n_samples, BEST_CONFIG['batch_size']):
            X_batch = X_train[i:i+BEST_CONFIG['batch_size']]
            y_batch = y_train[i:i+BEST_CONFIG['batch_size']]
            
            y_pred = nn.feedforward(X_batch)
            nn.backpropagation(y_batch, y_pred, criterion.derivative, optimizer)
            optimizer.config["timestep"] += 1
        
       
        train_preds = nn.feedforward(X_train)
        train_accuracy = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1))
        wandb.log({"epoch": epoch, "train_accuracy": train_accuracy})
    
    # Final evaluation
    test_preds = nn.feedforward(X_test)
    test_accuracy = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Plot and log confusion matrix
    cm_fig = plot_confusion_matrix(y_test, test_preds)
    
    
    temp_file = "confusion_matrix.png"
    cm_fig.savefig(temp_file)
    
    wandb.log({
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.Image(temp_file, caption="Fashion MNIST Classification Results")
    })
    
    # Clean up
    plt.close(cm_fig)
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()
