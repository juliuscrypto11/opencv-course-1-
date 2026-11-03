# pylint:disable=no-member

import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')

star1 = cv.fillPoly(
    blank.copy(),
    [np.array([
        [200, 30], [235, 150], [370, 150], [260, 220], [300, 370],
        [200, 280], [100, 370], [140, 220], [30, 150], [165, 150]
    ], dtype=np.int32)],
    255
)

star2 = cv.fillPoly(
    blank.copy(),
    [np.array([
        [200, 70], [220, 140], [290, 140], [235, 185], [255, 270],
        [200, 220], [145, 270], [165, 185], [110, 140], [180, 140]
    ], dtype=np.int32)],
    255
)

cv.imshow('Star 1', star1)
cv.imshow('Star 2', star2)

# Bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(star1, star2)
cv.imshow('Bitwise AND', bitwise_and)

# Bitwise OR --> all regions
bitwise_or = cv.bitwise_or(star1, star2)
cv.imshow('Bitwise OR', bitwise_or)

# Bitwise XOR --> non-overlapping regions
bitwise_xor = cv.bitwise_xor(star1, star2)
cv.imshow('Bitwise XOR', bitwise_xor)

# Bitwise NOT
bitwise_not = cv.bitwise_not(star2)
cv.imshow('Star 2 NOT', bitwise_not)

cv.waitKey(0)
cv.destroyAllWindows()