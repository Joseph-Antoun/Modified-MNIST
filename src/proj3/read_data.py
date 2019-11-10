import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    #--------------------------------------------------------------------------
    # Load and pre-process the train set
    #--------------------------------------------------------------------------
    train_data  = "../../data/data_kaggle/train_max_x"
    train_images = pd.read_pickle(train_data)

    print(train_images.shape)
    
    #--------------------------------------------------------------------------
    # Save a couple of images to jpg to view them
    #--------------------------------------------------------------------------
    for i in range(train_images.shape[0]):

        data  = train_images[i]
        image = 'raw_img/train_img_%s.png' % i

        plt.imsave(image, data, cmap='gray')
        print(i)

        if i >=50:
            break


if __name__ == "__main__":
    main()

