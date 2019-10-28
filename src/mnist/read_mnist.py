from mnist import MNIST


def test_data_load(mndata, images, labels):
    index = 0
    image = mndata.display(images[index])
    label = labels[index]

    print(image)
    print(label)


def main():

    # Load the training dataset
    mndata = MNIST('../../data/MNIST/raw/')
    images, labels = mndata.load_training()

    # Test that the dataset was properly loaded
    test_data_load(mndata, images, labels)


if __name__ == "__main__":
    main()

