# Aim: Implementation of Point Processing image enhancement Operations in Spatial Domain.
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

# Find the maximum gray level pixel and its location
max_ins = 0
max_loc = (0, 0)

height, width = image.shape[:2]

for r in range(height):
    for c in range(width):
        ins = image[r, c]
        if ins > max_ins:
            max_ins = ins
            max_loc = (r, c)

print(f"Max gray level pixel loc: {max_loc}")


# Create and display the negative image
neg = max_ins - image
show_image(neg, 'Negative image')


# Apply thresholding to create a binary image
threshold = 127.5
thresholded_image = (image > threshold) * 255
show_image(thresholded_image, "Thresholded Image")


# Apply contrast stretching
img2 = image.copy()

s1, s2 = 63.75, 127.5
r1, r2 = np.min(image) + 1, np.max(image)
l = max_ins

alpha = s1 / r1
beta = (s2 - s1) / (r2 - r1)
gamma = (l - 1 - s2) / (l - 1 - r2)

for r in range(height):
    for c in range(width):
        if img2[r, c] <= r1:
            img2[r, c] = alpha * img2[r, c]
        elif img2[r, c] <= r2 and img2[r, c] > r1:
            img2[r, c] = beta * (img2[r, c] - r1) + s1
        else:
            img2[r, c] = gamma * (img2[r, c] - r2) + s2

# Display the original image and the contrast stretched image
show_image(image)
show_image(img2)


# Gray level slicing without and with background
min_t = 63.75
max_t = 127.5
highlight_value = 255

# Create a mask for the highlighted region
mask = np.logical_and(image >= min_t, image <= max_t)

# Create copies of the original image
wb_without_bg = image.copy()
wb_with_bg = image.copy()

# Apply the highlight value to the pixels within the mask (without background)
wb_without_bg[mask] = highlight_value
show_image(wb_without_bg, 'Without Background')

# Apply a different value to the background pixels (with background)
wb_with_bg[~mask] = 50
show_image(wb_with_bg, 'With Background')

# Bit plane slicing and display
for bp in range(8):
    bit_plane = np.bitwise_and(image, 2 ** bp)
    nbp = (bit_plane * 255).astype(np.uint8)
    show_image(nbp, f"Bit Plane {bp + 1}")