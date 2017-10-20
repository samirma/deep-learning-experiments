# Author:kemo

import os
from PIL import Image
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import collections, numpy

np.random.seed(1337)

height = 54 
width = 72


def load_data(tol_num,train_num):

    data = np.empty((tol_num, width, height, 3),dtype="float32")
    
    labels = []

    # data dir
    imgs = os.listdir("data")
    
    for i in range(tol_num):
        # load the images and convert them into gray images
        img = get_image_from_file("data/"+imgs[i])

        arr = np.asarray(img, dtype="float32")
        try:
            category = imgs[i].split('.')[0].split('-')[0]
            
            if "135983" == category or "135979" == category:
                labels.append(category)
                data[i,:,:,:] = arr
        except:
            pass

    # the data, shuffled and split between train and test sets
    
    print(collections.Counter(labels))
    
    labels = one_hot_encoded(labels)
    tol_num = len(labels)
    rr = [i for i in range(tol_num)] 
    random.shuffle(rr)
    X_train = data[rr][:train_num]
    y_train = labels[rr][:train_num]
    X_test = data[rr][train_num:]
    y_test = labels[rr][train_num:]
    
    return (X_train,y_train),(X_test,y_test)

def get_image_from_file(path_img):
    img = Image.open(path_img)
    return pre_process_image(img)


def pre_process_image(img):
    #img = img.convert('L')
    return img


def one_hot_encoded(array_data):
    #values = array(array_data)
    #print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(array_data)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
