# Name: Anirbaan Ghatak
# Roll no.: C026
# Aim: Write a program to detect edges in the image using Robert, Prewitt and Sobel operators.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.resize(cv2.imread('IMG_2458_grey_CLOSEUP.jpg',
                cv2.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5)
nat = cv2.resize(cv2.imread('nature.jpg', cv2.IMREAD_GRAYSCALE),
                (0, 0), fx=0.5, fy=0.5)
medical = cv2.resize(cv2.imread(
    'medical.jpg', cv2.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5)


def edges(img):
    # Apply the Roberts operator
    edges_roberts = cv2.Sobel(
        img, cv2.CV_8U, 1, 0, ksize=3) + cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

    # Apply the Sobel operator
    edges_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3) + \
        cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

    # Apply the Prewitt operator
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    edges_prewitt = cv2.filter2D(
        img, -1, kernelx) + cv2.filter2D(img, -1, kernely)

    plt.figure(figsize=(10, 8))

    # Plot the first image in the top left position
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Plot the second image in the top right position
    plt.subplot(2, 2, 2)
    plt.imshow(edges_roberts, cmap='gray')
    plt.title('Roberts Edges')
    plt.axis('off')

    # Plot the third image in the bottom left position
    plt.subplot(2, 2, 3)
    plt.imshow(edges_sobel, cmap='gray')
    plt.title('Soble Edges')
    plt.axis('off')

    # Plot the fourth image in the bottom right position
    plt.subplot(2, 2, 4)
    plt.imshow(edges_prewitt, cmap='gray')
    plt.title('Prewitt Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


edges(img)
edges(nat)
edges(medical)
