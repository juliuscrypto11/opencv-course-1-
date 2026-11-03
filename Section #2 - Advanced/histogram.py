# pylint:disable=no-member

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('../Resources/Photos/atlas.webp')

# Check if image loaded successfully
if img is None:
    print("Error: Could not load image. Check the file path and filename.")
    exit()

cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Create circular mask
mask = cv.circle(
    blank,
    (img.shape[1] // 2, img.shape[0] // 2),
    100,
    255,
    -1
)

cv.imshow('Mask', mask)

# Apply mask
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

# Grayscale histogram
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])


# Colour Histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

colors = ('b', 'g', 'r')

for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)

plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()