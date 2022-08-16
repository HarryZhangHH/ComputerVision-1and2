import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
import uiux

# Note that these variables are lists of list since we want to be able process a set of images
images = []
images_gray = []
keypoints = []
descriptors = []
matches = []

# SIFT detector
def sift(n_features = 0, n_octaveLayers = 3, contrast_threshold = 0.04, edge_threshold = 10, sigma = 1.6):
    print("Find keypoints using SIFT.")

    try:
        if not images or not images_gray:
            raise Exception("Initialize the image sets images = [] and images_gray = [] inside the keypointmatching module with the image data!")
    except Exception as error:
        _, _, exc_tb = sys.exc_info()
        print(f"Error at line {exc_tb.tb_lineno}: {error}")
        exit()

    sift = cv.xfeatures2d.SIFT_create(n_features, n_octaveLayers, contrast_threshold, edge_threshold, sigma)

    # For each image detect features, i.e. keypoints, and their descriptor using the SIFT detector
    for i in range(len(images)):
        kp, ds = sift.detectAndCompute(images_gray[i], None)
        keypoints.append(kp)
        descriptors.append(ds)

    uiux.print_succes() 

# Brute force matching
def bfm(n_matching_points_to_connect = 10, n_draw_mode = "random", visualize = False):
    print("Find matches using brute force matching (BFM).")
    global matches

    try:
        # Create BFMatcher with default params
        bf = cv.BFMatcher()

        # Match descriptors between each pair of image in the set
        for i in range(len(images)):
            # Prevent out of bounds
            if i + 1 == len(images):
                break;

            # Match descriptors
            mtchs = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)

            # Apply ratio test
            good = []
            for m,n in mtchs:
                if m.distance < 0.75*n.distance:
                    good.append([m])

            # Save for use in other files
            matches.append(good)

            if n_matching_points_to_connect > len(good):
                raise Exception("<n_matching_points_to_connect> exceeds number of matching points found.")

            # According to parameter given draw a specific way from the found matches
            if n_draw_mode is "first":
                connected = good[:n_matching_points_to_connect]
            elif n_draw_mode is "last":
                connected = good[len(good) - n_matching_points_to_connect:len(good)]
            # Default: randomly draw n matching pairs
            else:
                connected = np.random.choice(np.array(good).ravel(), n_matching_points_to_connect, replace = False).reshape(n_matching_points_to_connect, 1)

            uiux.print_succes()
            print(f"{len(matches[i])} matching pair(s) found!")
            
            # cv.drawMatchesKnn expects list of lists as matches.
            if visualize:
                img = cv.drawMatchesKnn(images_gray[i], keypoints[i], images_gray[i + 1], keypoints[i + 1], connected, None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.title(f"Matches of {n_matching_points_to_connect} {n_draw_mode} feature descriptors between image {i + 1} and {i + 2}")
                plt.imshow(img)

        plt.show() 
    except Exception as error:
        _, _, exc_tb = sys.exc_info()
        print(f"Error at line {exc_tb.tb_lineno}: Something went wrong using BFM to match images!") 
        print(f"{error}")
        exit()