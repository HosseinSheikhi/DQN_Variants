"""
Created on Wed May  6 17:34:42 2020
    ObservationProcessing class will receive the raw image of Breakout
    1- will crop the unnecearry boarder (Header) for two reasons:
        the break out image header contains no information to help learning
        will get a square image
    2- convert image to gray (Based on deepmindpaper)
    3- convert size to 84*84 (Based on deepmindpaper)

@author: hossein
"""
import cv2
import matplotlib.pyplot as plt

TARGET_SIZE = 84


def process(raw_image):
    cropped_image = raw_image[34:-16, :, :]  # result would be a square image by size 160*160
    gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    final_image = cv2.resize(gray_img, (TARGET_SIZE, TARGET_SIZE))
    """
    plt.figure()
    plt.imshow(final_image)
    cv2.imshow("gray",final_image)
    cv2.waitKey(20)
    """
    return final_image