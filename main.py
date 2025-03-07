import cv2 as cv
import numpy as np

# Load the grayscale image
img = cv.resize(cv.imread("temple.jpg", cv.IMREAD_GRAYSCALE), (1004, 1004))

# Define the mosaic images
i_list = [
    cv.imread(f"Block{n}.png", cv.IMREAD_GRAYSCALE)
    for n in range(1, 5)
]

# Get image dimensions
img_height, img_width = img.shape
tile_height, tile_width = 2, 2

# Compute Canny edges once
can = cv.Canny(img, 0, 50)

# Function to choose the best matching image based on edge intensities
def choose(y0, y1, x0, x1):
    region = can[y0:y1, x0:x1]
    h_mid, w_mid = (y1 - y0) // 2, (x1 - x0) // 2

    sum1 = np.sum(region[:h_mid, :w_mid])  # Top-left
    sum2 = np.sum(region[:h_mid, w_mid:])  # Top-right
    sum3 = np.sum(region[h_mid:, :w_mid])  # Bottom-left
    sum4 = np.sum(region[h_mid:, w_mid:])  # Bottom-right

    sums = [sum1 + sum2, sum3 + sum4, sum1 + sum3, sum2 + sum4]
    return i_list[np.argmax(sums)]  # Select the image with the highest sum

# Process the image in tiles
for y in range(0, img_height, tile_height):
    for x in range(0, img_width, tile_width):
        selected_tile = choose(y, y + tile_height, x, x + tile_width)
        img[y:y + tile_height, x:x + tile_width] = 255 - selected_tile

# Display the result
cv.imshow('result', img)
cv.imshow('canny', can)
cv.waitKey(0)
cv.destroyAllWindows()
