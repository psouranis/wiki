import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import itertools
import operator
import os, csv
import tensorflow as tf

import time, datetime

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_values = [Sky,Building,Pole,Road_marking,Road,Pavement,Tree,
                SignSymbol,Fence,Car,Pedestrian,Bicyclist,Unlabelled]


import cv2                  
from keras.preprocessing.image import *
from tqdm import tqdm_notebook, tnrange
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_images(img_len,img_wid, img_path):
    imgs = []
    for img in tqdm(os.listdir(img_path)):
        path = os.path.join(img_path,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_wid,img_len))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgs.append(np.array(img))
    return(imgs)




def image_preprocessing(X_train,X_train_masks,X_val,X_val_masks,img_size,
                       img_path,mask_path,val_path,val_mask_path):
    data_gen_args = dict(rescale=1./255,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    zoom_range=0.15,
                    horizontal_flip=True)
    
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    val_datagen = ImageDataGenerator(**data_gen_args)
    val_masks =  ImageDataGenerator(**data_gen_args)
    
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(X_train_masks, augment=True, seed=seed)

    #Use different seed for validation
    seed_val = 7
    val_datagen.fit(X_train, augment=True, seed=seed_val)
    val_masks.fit(X_train_masks, augment=True, seed=seed_val)
    

    image_generator = image_datagen.flow_from_directory(img_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(mask_path,
    batch_size =1,
    class_mode=None,
    target_size=img_size,
    seed=seed)

    val_generator = val_datagen.flow_from_directory(val_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed_val)

    val_mask_generator = val_masks.flow_from_directory(val_mask_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed_val)
    
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    val_generator =zip(val_generator,val_mask_generator)
    
    return (train_generator,val_generator)

def plot_predictions(X_test,preds,path):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(X_test,aspect="auto")
    ax[0].set_title("Input")
    ax[1].imshow(preds, aspect="auto")
    ax[1].set_title("Prediction")
    fig.tight_layout()    
    
    

def equalize_hist(img):
    """Normalize each channel of the picture
    # Arguments 
        img : 3D array
    
    # Return Normalized img"""
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])

    return img



def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    print('Start One-Hot Vectorizing...')
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    print('Done..')
    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x




def median_frequency_balancing(image_files, num_classes=len(label_values)):
    label_to_frequency_dict = {}
    for i in range(num_classes):
        label_to_frequency_dict[i] = []

    for n in range(image_files.shape[0]):
        image = image_files[n]

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in range(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
#     class_weights[-1] = 0.0

    return class_weights