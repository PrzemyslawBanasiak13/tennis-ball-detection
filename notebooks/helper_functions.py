import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image
from PIL.ExifTags import TAGS


def plot_images(image_folder="data", img_count=4):
    """
    Plot <img_count> random images from <folder_name> from subfolders.
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
    Plot losses from <history> generated during training.
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


def print_image_info(image_path):
    """
    Prints informations about image from <image_path>.
    """
    # read the image data using PIL
    image = Image.open(image_path)

    # extract other basic metadata
    info_dict = {
    "Filename": image.filename,
    "Image Size": image.size,
    "Image Height": image.height,
    "Image Width": image.width,
    "Image Format": image.format,
    "Image Mode": image.mode,
    "Image is Animated": getattr(image, "is_animated", False),
    "Frames in Image": getattr(image, "n_frames", 1)
    }

    for label,value in info_dict.items():
        print(f"{label:25}: {value}")

    # extract EXIF data
    exifdata = image.getexif()

    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
