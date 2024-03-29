"""
    Extract vector feature and call database class to save
    Can you fix the def to make a databse vector feature out of all images
"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from os.path import join
from os import listdir
import numpy as np
from database import DBManagement


image_paths = list()
processed_img = dict()


def get_file_path():
    """ find the paths to all images in the traing set """
    path = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    image_folder = 'Flicker8k_Dataset/'
    """
        I process all image instead of only training.
        But I havent train on colab yet.
    """
    # file = open(path, 'r')
    file = listdir(image_folder)
    for line in file:
        abs_path = join(image_folder, line)
        image_paths.append(abs_path)


def get_vector_feature():
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    # preprocess all images to 299x299 size
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        # flatten
        features = np.reshape(features, 2048)

        file_name = image_path.split("/")[-1]
        processed_img[file_name] = features


def dump_data():
    db = DBManagement()
    db.save_image_data(processed_img)


if __name__ == '__main__':

    get_file_path()
    get_vector_feature()
    dump_data()

    #db = DBManagement()
    #print(len(db.data))
