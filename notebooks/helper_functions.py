import matplotlib.pyplot as plt
import numpy as np
import random
import os


def plot_images(image_folder="data", img_count=4):
    """
    Plot <img_count> random images from <folder_name> from subfolders
    """
    all_images = []
    axes = []
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    for image in os.listdir(image_folder):
        all_images.append(os.path.join(image_folder, image))

    for i in range(img_count):
        im_path = random.choice(all_images)
        im = plt.imread(im_path)
        axes.append(fig.add_subplot(1, img_count, i + 1))
        subplot_title = f"Image {im_path}"
        axes[-1].set_title(subplot_title)
        plt.imshow(im)

    fig.tight_layout()
    plt.show()


def plot_losses(history):
    """
    Plot losses from <history> generated during training
    """
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('train and valid loss vs epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
