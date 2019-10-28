import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim




def save2jpg(img):

    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("view_classification.png")


def load_preprocess(data_dir, train=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(
        data_dir, 
        download=True, 
        train=train, 
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return data_loader



def main():

    #--------------------------------------------------------------------------
    # Load and pre-process the train set
    #--------------------------------------------------------------------------
    train_data   = "../../data/data_kaggle/train_max_x"
    train_loader = torch.load(train_data)
    sys.exit(0)
    
    #--------------------------------------------------------------------------
    # Save a couple of images to jpg to view them
    #--------------------------------------------------------------------------
    images, labels = next(iter(train_loader))
    img = images[0].view(1, 784)
    print(img)



if __name__ == "__main__":
    main()

