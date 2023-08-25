import os 
import time
import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.utils import load_img, img_to_array

def read_data_file(directory, file_path):
    
    """
    Reads a CSV file from the specified directory and file path.
    
    Parameters:
        directory (str): The directory where the CSV file is located.
        file_path (str): The relative path to the CSV file.
        
    Returns:
        pd.DataFrame: The DataFrame containing the read data.
    """
    # Get the current working directory
    cwd = os.getcwd()

    # Construct the absolute file path to the CSV file
    file_path = os.path.join(cwd, '..', directory, file_path)

    # Read the CSV file
    data = pd.read_csv(file_path)

    return data


def preprocess_data(train):
    """
    Performs one-hot encoding on categorical variables in the DataFrame.
    
    Parameters:
        train (pd.DataFrame): The DataFrame to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame with one-hot encoded columns.
    """
    # Perform one-hot encoding
    train_encoded = pd.get_dummies(train, columns=['race'])
    train_encoded = pd.get_dummies(train_encoded, columns=['gender'])
    train_encoded = pd.get_dummies(train_encoded, columns=['emotion'])
    
    return train_encoded


def image_standardization(image):
    """
    Perform image standardization by subtracting mean and dividing by standard deviation.
    
    Parameters:
        image (numpy.ndarray): Input image in float32 format.
        
    Returns:
        numpy.ndarray: Standardized image.
    """
    # Convert the image to float32 (assuming it's in uint8 format)
    image = image.astype(np.float32)

    # Mean subtraction
    mean = np.mean(image)
    image -= mean

    # Normalization
    std = np.std(image)
    image /= std

    return image

def read_image_data(train_encoded, directory, image_file_path):
    
    """
    Read and preprocess image data from specified directory.
    
    Parameters:
        train_encoded (pd.DataFrame): DataFrame containing image_id and labels.
        directory (str): Directory path for images.
        image_file_path (str): Subdirectory for image files.
        
    Returns:
        numpy.ndarray: Processed image data (X).
        numpy.ndarray: Corresponding labels (y).
    """
    
    
    labels = train_encoded.drop("image_id", axis=1)
    labels = labels.iloc[:, :14]
    missing_files = 0
    # Get the current working directory
    cwd = os.getcwd()
    
    # Construct the absolute file path to the image folder file
    image_path = os.path.join(cwd, '..', directory, image_file_path)
    
    train_image = []
    train_label = []
    count = 0
    for i in tqdm(range(train_encoded.shape[0])):
        try:
            img = load_img(image_path + train_encoded['image_id'][i], target_size=(224, 224, 3))
            img = img_to_array(img)
            img = img / 255
            train_image.append(image_standardization(img))
            train_label.append(labels.iloc[i])
            count = count + 1
        except Exception as e:
            missing_files += 1
            print(f"An error occurred while processing image {train_encoded['image_id'][i]}: {e}")

    
    #print("missing", missing_files)

    #print(len(train_image), len(train_label))
    X = np.array(train_image)
    y = np.array(train_label)
    
    return X, y

