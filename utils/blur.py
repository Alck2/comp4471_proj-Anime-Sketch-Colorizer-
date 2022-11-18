import cv2 as cv
from PIL import Image

def blur_img(img, kernel_size):
    return cv.blur(img, kernel_size)

