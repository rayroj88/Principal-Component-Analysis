import numpy as np
import cv2
import matplotlib.pyplot as plt


def pca_detect_digit(image, mean_digit, eigenvectors, N):
    """
    Detects the center of a digit in an image using PCA.

    Parameters:
    image (numpy.ndarray): The input image.
    mean_digit (numpy.ndarray): The mean digit image (28x28 array)
    eigenvectors (numpy.ndarray): The eigenvectors of the digit images.
    N (int): The number of eigenvectors to use.

    Returns:
    tuple: The center of the detected digit as a tuple of (row, column) coordinates.
    """
    #Set closest match to big number
    closest_match  = float('inf')
    
    #Set window to 28 x 28
    for i in range(image.shape[0] - 28 + 1):
        for j in range(image.shape[1] - 28 + 1):
            window = image[i:i+28, j:j+28]
            
            #flatten the window
            window_vector = window.reshape(-1)
                
            #normalize the window
            window_vector_centered = (window_vector - np.mean(window_vector)) / np.std(window_vector)
            
            #dot product of window and eigenvectors
            projected_vector = np.dot(eigenvectors[:, :N].T, window_vector_centered)
            
            # Add the mean back into the vector
            window_reconstruction = np.dot(projected_vector, eigenvectors[:, :N].T) + mean_digit.flatten()
            
            #Take the norm to calculate the error
            error = np.linalg.norm(window_vector_centered - window_reconstruction)
            
            #Check for lowest error
            if error < closest_match:
                closest_match = error
                detection_center = (i + 14,j + 14)

    return detection_center
