import numpy as np
import cv2
import os

def get_eigenvectors(data_file_path, digit):
    """
    Computes the mean vector and eigenvectors of the covariance matrix of a given digit in a dataset.

    Args:
        data_file_path (str): The path to the dataset file.
        digit (int): The digit to extract from the dataset.

    Returns:
        tuple: A tuple containing the mean vector and eigenvectors of the covariance matrix.
    """
    #Create an array to store our vectors from the images
    vectors = []
    
    #Pull the values from csv into a 2d array
    vectors = np.loadtxt(data_file_path, delimiter=",")
    
    #Collect only vectors that start in digit
    vectors = vectors[vectors[:,0] == digit]
    
    #Remove first column from vector list
    vectors = vectors[:, 1:]
     
    #Normalize vectors to have a zero mean and unit variance
    for i in range(vectors.shape[0]):
        vectors[i,:] = (vectors[i,:] - np.mean(vectors[i,:])) / np.std(vectors[i,:])
    
    #Calculate mean vector
    mean_vector = np.mean(vectors, axis=0)
    
    #Compute Covariance Matrix
    vectors = vectors - mean_vector
    covariance_matrix = np.cov(vectors, rowvar = False)
    
    #Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    #Sort by Eigenvalues to get descending order
    idx = np.argsort(eigenvalues)
    
    #Reverse Eigenvalue order
    idx = idx[::-1]
    
    #Sort eigenvectors by eigenvalue order
    eigenvectors = eigenvectors[:, idx]
            
    return mean_vector, eigenvectors