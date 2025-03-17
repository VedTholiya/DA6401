import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb



wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
wandb.init(project="DA6401 - Assignment 1", entity='ma23c047-indian-institute-of-technology-madras')  #wandb login

#load images

(x_train, y_train), (test_images, test_labels) = fashion_mnist.load_data()   

# Plot images
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fig, ax = plt.subplots(2,5)
ax = ax.flatten()
for i in range(10):
    index = np.argwhere(y_train == i)[0]
    sample = np.reshape(x_train[index], (28, 28))
    ax[i].imshow(sample)
    wandb.log({"Sample Images": [wandb.Image(sample, caption=CLASS_NAMES[i])]})
wandb.finish()