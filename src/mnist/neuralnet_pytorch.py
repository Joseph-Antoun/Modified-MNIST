import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

from mnist import MNIST


#
# From https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
#


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig("view_classification.png")


def classification_to_image(model, img):

    with torch.no_grad():
        logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        print("Predicted Digit =", probab.index(max(probab)))
        view_classify(img.view(1, 28, 28), ps)


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
    train_dir       = "../../data/"
    train_loader    = load_preprocess(train_dir, train=True)
    
    #--------------------------------------------------------------------------
    # Build the neural network
    #--------------------------------------------------------------------------
    input_size   = 784
    hidden_sizes = [128, 64]
    output_size  = 10
    
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    print(model)

    #--------------------------------------------------------------------------
    # Train the neural network
    #--------------------------------------------------------------------------
    criterion = nn.NLLLoss()
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)
    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL loss

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
    
            # Training pass
            optimizer.zero_grad()
    
            output = model(images)
            loss = criterion(output, labels)
    
            #This is where the model learns by backpropagating
            loss.backward()
    
            #And optimizes its weights here
            optimizer.step()
    
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
            print("\nTraining Time (in minutes) =",(time()-time0)/60)


    #--------------------------------------------------------------------------
    # Test its performance - one classification
    #--------------------------------------------------------------------------

    # Load the test set
    val_loader = load_preprocess(train_dir, train=False)

    # Save the output of one classification as a png image
    images, labels = next(iter(val_loader))
    img = images[0].view(1, 784)
    classification_to_image(model, img)

    # Performance on the entire test set
    correct_count, all_count = 0, 0

    for images,labels in val_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            if(true_label == pred_label):
                correct_count += 1
            all_count += 1


    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


if __name__ == "__main__":
    main()

