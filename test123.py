# Aim: Write a program to enhance the quality of an image by noise removal
# Name: Anirbaan Ghatak
# Roll No.: C026

import cv2
import numpy as np

def show_image(image, title='image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'IMG_2458.jpg'
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (500, 500))
show_image(image)

def salt_pepper(image):

    noisy_imagesp = np.copy(image)
    num_pixels = int(0.02 * image.size)

    # Add salt noise (white pixels)
    salt_coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy_imagesp[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy_imagesp[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_imagesp


#adding noise to the image
def addnoise(image):
    mean = 0
    stddev = 180
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    noisy_img = cv2.add(image, noise)
    return noisy_img

def high_pass_filter(noisy_img, kernel_size = 3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    blurred = cv2.filter2D(noisy_img, -1, kernel)
    hpf = noisy_img - blurred
    
    return hpf

def low_pass_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    lpf = cv2.filter2D(image, -1, kernel)

    return lpf


def median_filter(noisy_img, kernel_size=3):
    median_filter = cv2.medianBlur(noisy_img, kernel_size)
    return median_filter

noisy_image = addnoise(image)
show_image(noisy_image, 'Noisy Image')
show_image(high_pass_filter(noisy_image, 2), 'High Pass Filter')
show_image(low_pass_filter(noisy_image), 'Low Pass Filter')
show_image(median_filter(noisy_image), 'Median Filter')

sp_img = salt_pepper(image)
show_image(sp_img, 'Salt&Pepper Image')
show_image(high_pass_filter(sp_img, 3), 'Salt&Pepper High Pass Filter')
show_image(low_pass_filter(sp_img), 'Salt&Pepper Low Pass Filter')
show_image(median_filter(sp_img), 'Salt&Pepper Median Filter')