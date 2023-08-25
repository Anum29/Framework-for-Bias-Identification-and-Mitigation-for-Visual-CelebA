import pandas as pd
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array



def train_val_test_split(X, y):
    """
    Split the data into training, validation, and test sets.

    Args:
        X: Input features.
        y: Labels.

    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    # Split the data into training, validation, and test sets using train_test_split
    # First, split into training and a temporary set (X_temp, y_temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # Next, split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, test_size=0.2)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# Function to build and train the model
def build_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Build and train a deep learning model.

    Args:
        model_name (str): Name of the model architecture.
        X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and test data.

    Returns:
        Model: Trained model.
    """
    learning_rate = 0.0001
    batch_size = 64
    epochs = 1
    
    if model_name == 'vggnet':
        vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in vgg16_base.layers:
            layer.trainable = False
        _flat = Flatten()(vgg16_base.output)
        _dense1 = Dense(4096, activation="relu")(_flat)
        _dropout1 = Dropout(0.5)(_dense1)
        _dense2 = Dense(4096, activation="relu")(_dropout1)
        _dropout2 = Dropout(0.5)(_dense2)
        _output = Dense(14, activation="softmax")(_dropout2)
        vgg16_model = Model(inputs=vgg16_base.input, outputs=_output)

        optimizer = Adam(learning_rate=learning_rate)
        vgg16_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC()
        ])
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        history = vgg16_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
        vgg16_model.save('my_model.h5')
        return vgg16_model
    
    elif model_name == 'resnet':
        # Load the ResNet50 model with pretrained ImageNet weights
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the layers in the base ResNet50 model
        for layer in resnet_base.layers:
            layer.trainable = False

        # Create the custom classification head
        _flat = Flatten()(resnet_base.output)
        _dense1 = Dense(4096, activation="relu")(_flat)
        _dropout1 = Dropout(0.5)(_dense1)
        _dense2 = Dense(4096, activation="relu")(_dropout1)
        _dropout2 = Dropout(0.5)(_dense2)
        _output = Dense(14, activation="softmax")(_dropout2)

        # Create the final model with the base ResNet50 and custom classification head
        resnet_model = Model(inputs=resnet_base.input, outputs=_output)

        # Compile the model using the Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)  
        resnet_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train the model with early stopping
        #history = vgg16_model.fit(X_train, y_train, batch_size=64, epochs=120, validation_data=(X_val, y_val))

        # Train the model with early stopping
        history = resnet_model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
        resnet_model.save('my_model.h5')
        return resnet_model
    
    elif model_name == 'googlenet':
        # Load the InceptionV3 model with pretrained ImageNet weights
        inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the layers in the base InceptionV3 model
        for layer in inception_base.layers:
            layer.trainable = False

        # Create the custom classification head
        _flat = Flatten()(inception_base.output)
        _dense1 = Dense(4096, activation="relu")(_flat)
        _dropout1 = Dropout(0.5)(_dense1)
        _dense2 = Dense(4096, activation="relu")(_dropout1)
        _dropout2 = Dropout(0.5)(_dense2)
        _output = Dense(14, activation="softmax")(_dropout2)

        # Create the final model with the base InceptionV3 and custom classification head
        inception_model = Model(inputs=inception_base.input, outputs=_output)

        # Compile the model using the Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)  # You can adjust the learning rate as needed
        inception_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train the model with early stopping
        #history = vgg16_model.fit(X_train, y_train, batch_size=64, epochs=120, validation_data=(X_val, y_val))

        # Train the model with early stopping
        history = inception_model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
        return inception_model
    
    

# Function to refit the model with generated data
def refit_model(X_generated, y_generated, X_val, y_val):
    """
    Refit a trained model with generated data.

    Args:
        X_generated, y_generated, X_val, y_val: Generated data and validation data.

    Returns:
        Model: Refitted model.
    """
    learning_rate = 0.0001
    batch_size = 64
    epochs = 10
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    loaded_model = load_model('my_model.h5')
    history1 = loaded_model.fit(X_generated, y_generated, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return loaded_model


# Function to evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model.
        X_test, y_test: Test data.

    Returns:
        tuple: Test accuracy, precision, and AUC.
    """
    test_loss, test_accuracy, precision, auc = model.evaluate(X_test, y_test)
    print("\nTest Accuracy:", test_accuracy)
    print("\nTest loss:", test_loss)
    print("\nPrecision:", precision)
    print("\nAuc:", auc)
    return test_accuracy, precision, auc


# Function to make predictions and analyze attribute accuracy
def make_predictions(model, X_test, y_test, labels):
    """
    Make predictions using a trained model and analyze attribute accuracy.

    Args:
        model: Trained model.
        X_test, y_test: Test data.
        labels: DataFrame with attribute labels.

    Returns:
        tuple: List of failed samples, attribute accuracies, and attribute precisions.
    """
    y_pred = model.predict(X_test)
    num_attributes = 14
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    attribute_accuracies = {}
    attribute_precisions = {}

    for i in range(num_attributes):
        y_pred_attribute = y_pred_binary[:, i]
        y_test_attribute = y_test[:, i]

        attribute_accuracy = accuracy_score(y_test_attribute, y_pred_attribute)
        attribute_precision = precision_score(y_test_attribute, y_pred_attribute)

        attribute_accuracies[f'Attribute_{labels.columns[i]}'] = attribute_accuracy * 100
        attribute_precisions[f'Attribute_{labels.columns[i]}'] = attribute_precision * 100

    failed_samples = []
    for i in range(len(y_pred)):
        y_pred_attribute = y_pred_binary[i]
        y_test_attribute = y_test[i]
        attribute_accuracy = accuracy_score(y_test_attribute, y_pred_attribute)

        if attribute_accuracy < 0.87:
            failed_samples.append(i)
    
    return failed_samples, attribute_accuracies, attribute_precisions


# Function to analyze and group failed samples
def analyze_failed_samples(failed_samples, merged_df):
    """
    Analyze and group failed samples based on attributes.

    Args:
        failed_samples: List of indices of failed samples.
        merged_df: Merged DataFrame containing attribute information.

    Returns:
        DataFrame: Grouped and sorted failed samples.
    """
    df = pd.DataFrame(columns=['race', 'gender', 'emotion', 'other_attributes'])
    index_column_number = 5  # The index of the desired column
    count = 0

    for j in range(len(failed_samples)): 
        n = failed_samples[j]
        row = merged_df.iloc[n]
        selected_attributes = row[row == 1].index

        sentence = "" + ", ".join(selected_attributes)
        df.loc[count, df.columns[0]] = row[1]
        df.loc[count, df.columns[1]] = row[2]
        df.loc[count, df.columns[2]] = row[3]
        df.loc[count, df.columns[3]] = sentence

        count = count + 1
        
    grouped = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

    sorted_grouped = grouped.sort_values('count', ascending=False)

    sorted_grouped['race'] = sorted_grouped['race'].str.replace('race_', '')
    sorted_grouped['gender'] = sorted_grouped['gender'].str.replace('gender_', '')
    sorted_grouped['emotion'] = sorted_grouped['emotion'].str.replace('emotion_', '')

    return sorted_grouped


# Function to save missclassified samples
def save_over_sampling_data(model_name, directory, missclassified_samples):
    """
    Save missclassified samples to a CSV file.

    Args:
        model_name (str): Name of the model.
        directory (str): Directory to save the file.
        missclassified_samples: DataFrame containing missclassified samples.
    """
    cwd = os.getcwd()
    file_name = f"miss_classified_image_samples_{model_name}.csv"
    file_path = os.path.join(cwd, '..', directory, file_name)
    missclassified_samples.to_csv(file_path, index=False)