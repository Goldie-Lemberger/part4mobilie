import math

try:
    import test
    import os
    import json
    import glob
    import argparse
    import cv2

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc
    from PIL import Image
    import matplotlib.pyplot as plt
    from scipy.spatial import distance

except ImportError:
    print("Need to fix the installation")
    raise


def build_kernel():
    """The kernel that is built is an image of 13 pixels on 13 pixeles.
    The image  is a blurred white circle in the center surrounded by a black square that creates a strong contrast,
     that will be suited for detecting a traffic light.
     It will be looking for a circle shaped object with a strong contrast frame
     :return Kernel: 2D array with sum 0
     """
    circle_img = Image.open("light.png").convert('L')
    kernel = np.asarray(circle_img)
    kernel = kernel.astype(np.float32)
    kernel -= 100
    sum_circle = np.sum(kernel)
    area = circle_img.width * circle_img.height
    kernel -= (sum_circle / area)
    max_kernel = np.max(kernel)
    # we want to keep the kernel's values as float so we divide it by it's max
    kernel /= max_kernel
    return kernel


def find_tfl_lights(c_image: np.ndarray, kernel, some_threshold):
    """    this function receives the image that we will be searching on and the current used kernel
    will return the coordinates in the image of all traffic lights
    :return - tuple of X and Y values
    """

    # get's the red layer of the picture and the green layer of the picture
    red_matrix, green_matrix = np.array(c_image)[:, :, 0], np.array(c_image)[:, :, 1]

    new_red = sg.convolve(red_matrix, kernel, mode='same')
    new_green = sg.convolve(green_matrix, kernel, mode='same')
    # filters to get the max match in each area of the green and red after doing the convolvation
    red_max = maximum_filter(new_red, size=300)
    green_max = maximum_filter(new_green, size=300)
    red_max_point = red_max == new_red
    green_max_point = green_max == new_green

    y_red, x_red = np.where(red_max_point)
    y_green, x_green = np.where(green_max_point)



    min_distance=150
    for index in range(len(x_red)):
        suc=False
        for index2 in range(index+1,len(x_red)):
            point1 = np.array((x_red[index],y_red[index]))
            point2 = np.array((x_red[index2],y_red[index2]))
            dist = np.linalg.norm(point1 - point2)
            # print(dist)
            if dist<min_distance:
                suc=True
        if y_red[index] < 40 or y_red[index] > 1000:
            suc = True
            # marking the unnecessary indexes as -1 in order to delete them
        if suc:
            x_red[index] = -1
            y_red[index] = -1
    y_red = np.delete(y_red, np.where(y_red == -1))
    x_red = np.delete(x_red, np.where(x_red == -1))
    # In assumption that there are no traffic light lower than 40 or higher than 1000 we will remove
    #  the green coordinates that are out of this range
    for index in range(len(x_green)):
        suc = False
        for index2 in range(index + 1, len(x_green)):
            point1 = np.array((x_green[index], y_green[index]))
            point2 = np.array((x_green[index2], y_green[index2]))
            dist = np.linalg.norm(point1 - point2)
            # print(dist)
            if dist < min_distance:
                suc = True
        if y_green[index] < 40 or y_green[index] > 1000:
            # marking the unnecessary indexes as -1 in order to delete them
            suc = True

        if suc:
            x_green[index] = -1
            y_green[index] = -1


    y_green = np.delete(y_green, np.where(y_green == -1))
    x_green = np.delete(x_green, np.where(x_green == -1))

    return x_red, y_red, x_green, y_green




def test_find_tfl_lights(image_path, kernel):
    """
    Run the attention code, Testing the founded coordinates of the traffic lights

    """
    # opening the image as cv
    img = cv2.imread(image_path)
    small_img = cv2.pyrDown(img)

    # finding the coordinates of the green and red traffic lights of the reduced image
    red_x_small, red_y_small, green_x_small, green_y_small = find_tfl_lights(small_img, kernel, some_threshold=42)
    image = np.array(Image.open(image_path))

    # finding the coordinates of the green and red traffic lights of the original image
    red_x_big, red_y_big, green_x_big, green_y_big = find_tfl_lights(image, kernel, some_threshold=42)

    # converting the returned coordinates to numpy type
    #  resizing the fixels of the detected cordinations of the reduced image
    # that the detected cordinates will be right on the original picture
    a1 = np.array(red_x_small * 2)
    a2 = np.array(red_y_small * 2)
    b1 = np.array(red_x_big)
    b2 = np.array(red_y_big)
    c1 = np.array(green_x_small * 2)
    c2 = np.array(green_y_small * 2)
    d1 = np.array(green_x_big)
    d2 = np.array(green_y_big)


    # concatenating the same type coordinates of the reduced image and the original one.
    red_x = np.concatenate([a1, b1])
    red_y = np.concatenate([a2, b2])
    green_x = np.concatenate([c1, d1])
    green_y = np.concatenate([c2, d2])


    return red_x, red_y, green_x, green_y, image
