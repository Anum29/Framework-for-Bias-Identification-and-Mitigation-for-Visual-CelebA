import pandas as pd
import numpy as np
import math
import os

# Function to compute Shannon Diversity and Evenness
def compute_shannon_diversity_and_evenness(p_values):
    """
    Computes Shannon Diversity and Evenness based on provided probability values.
    
    Parameters:
        p_values (numpy.ndarray): Array of probability values for each category.
        
    Returns:
        tuple: Shannon Diversity and Evenness.
    """
    S = len(p_values)
    H = -np.sum(p_values * np.log(p_values))
    E = H / np.log(S)
    return H, E

# Function to compute Simpson Diversity and Evenness
def compute_simpson_diversity_and_evenness(p_values):
    """
    Computes Simpson Diversity and Evenness based on provided probability values.
    
    Parameters:
        p_values (numpy.ndarray): Array of probability values for each category.
        
    Returns:
        tuple: Simpson Diversity and Evenness.
    """
    D = 1 / np.sum(p_values * p_values)
    E = D / len(p_values)
    return D, E

def compute_metrics(df):
    """
    Computes diversity and evenness metrics for each attribute in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing attributes.
        
    Returns:
        pd.DataFrame: A DataFrame with computed metrics for each attribute.
    """
    # List of attributes
    attributes = ['race', 'gender', 'emotion']

    # Initialize a list to store results
    results = []

    # Compute diversity and evenness for each attribute
    for attribute in attributes:
        p_values = df[attribute].value_counts(normalize=True).values
        shannon_diversity, shannon_evenness = compute_shannon_diversity_and_evenness(p_values)
        simpson_diversity, simpson_evenness = compute_simpson_diversity_and_evenness(p_values)

        result = {
            'Attribute': attribute,
            'Shannon Diversity': shannon_diversity,
            'Shannon Evenness': shannon_evenness,
            'Simpson Diversity': simpson_diversity,
            'Simpson Evenness': simpson_evenness
        }

        results.append(result)

    # Create a new DataFrame to store results
    results_df = pd.DataFrame(results)
    return results_df

def count_attributes(df):
    """
    Displays categories and counts for each attribute in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing attributes.
    """
    # List of attribute names
    attributes = ['race', 'gender', 'emotion']

    # Display categories and counts for each attribute
    for attribute in attributes:
        attribute_counts = df[attribute].value_counts()
        print(f"Categories and counts for {attribute}:")
        print(attribute_counts)
        print("\n")
