import numpy as np
from numpy import int8, linalg
from numpy.core.fromnumeric import argmax
from numpy.lib.index_tricks import r_
from numpy.random.mtrand import rand, random_sample
import sys
import matplotlib.pyplot as plt
import cv2 as cv
import uiux

# Contrained least squares method
def clsm(A):
    # Solve as an eigenvalue problem using singular value decomposition
    _, _, vh = np.linalg.svd(A)
    min = 8
    return vh[min]

# Contruct matrix A of knowns
def construct_A(pairs):
    """" 
        Let the input be an array containing the set of pairs of matching points, where pairs_i = [(x_i, y_i), (x_i', y_i')], with (x_i, y_i) <-> (x_i', y_i')
    """
    A = []

    for pair in pairs:
        p1 = pair[0]
        p1_x, p1_y = p1

        p2 = pair[1]
        p2_x, p2_y = p2

        x_eq_row = [p1_x, p1_y, 1, 0, 0, 0, -p2_x * p1_x, -p2_x * p1_y, -p2_x]
        y_eq_row = [0, 0, 0, p1_x, p1_y, 1, -p2_y * p1_x, -p2_y * p1_y, -p2_y]

        A.append(x_eq_row)
        A.append(y_eq_row)

    return np.array(A)

# Compute projection of point p given its coordinates
def project_p(H, x, y):
    p = np.transpose(np.atleast_2d([x, y, 1]))
    return p, np.matmul(H, p)

# Returns al the inliers given our model i.e. warping with a chosen homograpy
def get_inliers(H, matches, e):
    """ e: Error marge, i.e. radius of pixel location deviation from the true value. """
    inliers = []
    
    for match in matches:
        p1, p2 = match

        p1_x, p1_y = p1
        p2_x, p2_y = p2
        p1, p_est = project_p(H, p1_x, p1_y)

        p_est_x = p_est[0][0]
        p_est_y = p_est[1][0]

        # Classify it as inlier when it lies within a radius of <e> pixels compared to the coordinates of the matching point
        if abs(p2_x - p_est_x) <= e and abs(p2_y - p_est_y) <= e:
            inliers.append(p1)

    return inliers

# # Regularize boundaries, i.e. range/domain, such that it is never out of bounds
# def regularize_boundaries(boundaries, min, max):
#     for i, boundary in enumerate(boundaries):
#         print(f"{i}: {boundary}")
#         if boundary < min:
#             boundaries[i] = min
#         if boundary > max:
#             boundaries[i] = max
#     return boundaries

# Ransac algorithm
def ransac(T, kp1, kp2, N = 500, P = 50, e = 10):
    print("Applying RANSAC algorithm")

    try:
        if P < 4:
            raise Exception("To solve the system of linear equations RANSAC requires at least 4 mathing pairs. P >= 4.")

        if P > len(T):
            raise Exception(f"Number of matching pairs to be picked P = {P} exceeds number of matching points found. Choose a number below {len(T)}.")

    except Exception as error:
        _, _, exc_tb = sys.exc_info()
        print(f"Error at line {exc_tb.tb_lineno}: {error}")
        exit()

    matches = []

    # Get coordinate points of found matching pairs
    for match in T:
        p1 = kp1[match[0].queryIdx].pt
        p2 = kp2[match[0].trainIdx].pt
        matches.append([p1, p2])

    print(f"* Finding best model fit.")

    H_best = np.zeros((3, 3))
    inliers_max = 0

    # Repeat N times
    for i in range(N):
        # Random sample taken from the found matches pairs
        s_matches = []
        # Temp
        src_pts = []
        dst_pts = []

        # Pick P random matches given set T
        random_indices = np.random.choice(len(matches), P, replace = False)

        # Construct matching pairs coordinates set
        for index in random_indices:
            p1, p2 = matches[index]
            s_matches.append([p1, p2])
            src_pts.append(p1)
            dst_pts.append(p2)

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        # Solve for A^T*A*x = lambda * x
        A = construct_A(s_matches) 
        x = clsm(A)
           
        # Homography
        H = np.atleast_2d(x).reshape(3, 3)
        H = H / H[2, 2]
        inliers = get_inliers(H, matches, e)

        print(f"Iteration {i + 1} inliers found: {len(inliers)}")

        """ 
            When higher accuracy obtained: set new record, i.e. threshold, to select model
            and save newly found best model, i.e. homography
        """
        if len(inliers) > inliers_max:
            inliers_max = len(inliers)
            H_best = H

    # Force good model selection
    if inliers_max == 0:
        print(f"Could not find a good model inliers found is {inliers_max}. Try again with various P (P is now set to {P})!")
        exit()
    else:
        uiux.print_succes()
        print(f"Repeated model fitting with N = {N} (number of iterations).")
        print(f"Best model found with sample size P = {P} resulted in {inliers_max} inliers, corresponding to {inliers_max} out of {len(matches)} positive observations.")
        print(f"Accuracy = {(inliers_max/len(matches)) * 100}%.\nHomography matrix H as follows:\n\n{H_best}\n\n")

    return H_best
    