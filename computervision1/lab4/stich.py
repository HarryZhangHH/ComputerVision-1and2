from matplotlib.pyplot import imshow
import uiux
import keypoint_matching as kpm
import RANSAC as rnsc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

# Convert image in BGR color space to GRAY color space
def BGR2GRAY(input, output_type = "uint8"):
    image_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

    if type is "float32":
        return image_gray.astype(np.float32)
    elif type is "uint8":  
        return image_gray.astype(np.uint8)
    else:
        return image_gray.astype(np.uint8)

# Reads in an image
def read_image(img):
    try:
        print(f"Read in '{img}' image.")
        image = cv.imread(img)
  
        if image is None:
            raise Exception("File name/path might be wrong.")

        uiux.print_succes() 
        return image
    except Exception as error:
        uiux.print_error(f"Something went wrong reading the '{img}'' image. {error}")
        exit()

def get_bounding_box(H, img):
    y_max, x_max, _ = img.shape
    y_max -= 1 # Since we want the max indices
    x_max -= 1

    # Compute projection coordinates of corners of the input image
    _, tl = rnsc.project_p(H, 0, 0) # Top left corner
    _, tr = rnsc.project_p(H, x_max, 0) # Top right corner
    _, bl = rnsc.project_p(H, 0, y_max) # Bottom left corner
    _, br = rnsc.project_p(H, x_max, y_max) # Bottom right corner

    corners_x = np.array([tl[0][0], tr[0][0], bl[0][0], br[0][0]])
    corners_y = np.array([tl[1][0], tr[1][0], bl[1][0], br[1][0]])

    cx_min = np.amin(corners_x)
    cx_max = np.amax(corners_x)

    cy_min = np.amin(corners_y)
    cy_max = np.amax(corners_y)

    return np.array([[cy_min, cy_max], [cx_min, cx_max]])

# Estimate size of stitched images frame
def estimate_size_frame(H, img):
    print(f"* Estimating new frame size [y, x].")
  
    y_max, x_max, _ = img.shape

    y_boundaries, x_boundaries = get_bounding_box(H, img)
    cy_min, cy_max = y_boundaries
    cx_min, cx_max = x_boundaries

    delta_x_l, delta_x_r = 0, 0
    delta_y_t, delta_y_b = 0, 0
 
    # Extension of x-axis on the left side
    if cx_min < 0:
        delta_x_l = abs(round(cx_min))

    # Extension of x-axis on the right side
    if cx_max > x_max:
        delta_x_r = abs(round(cx_max - x_max))
        
    # Extension of y-axis on the top side
    if cy_min < 0:
        delta_y_t = abs(round(cy_min))

    # Extension of y-axis on the bottom side
    if cy_max > y_max:
        delta_y_b = abs(round(cy_max - y_max))

    shape = [delta_y_t + y_max + delta_y_b, delta_x_l + x_max + delta_x_r, 3] 
    uiux.print_succes()
    print(f"old shape: {list(img.shape)}")
    print(f"new shape: {shape}")
    print(f"dy = {delta_y_t}, dx = {delta_x_l}\n")
    return shape, delta_y_t, delta_x_l

# Forward warping
def forward_warping(H, img, frame, dx = 0, dy = 0, nearest_neighbor = True):
    r, c, _ = img.shape
    r_frame, c_frame, _ = frame.shape

    img_out = np.empty(img.shape)

    for y_i in range(r):
        for x_i in range(c):
            # Value in input image
            val = img[y_i][x_i]

            _, p_new = rnsc.project_p(H, x_i, y_i)
            p_new_x = p_new[0][0]
            p_new_y = p_new[1][0]

            if nearest_neighbor:
                p_new_x = round(p_new_x)
                p_new_y = round(p_new_y)
            
            # Prevent out of bounds when projection falls out of frame
            x_i_frame = p_new_x + dx
            y_i_frame = p_new_y + dy

            if not ((x_i_frame >= c_frame or y_i_frame >= r_frame) or (x_i_frame < 0 or y_i_frame < 0)):
                # TRY TO PLACE IN FRAME
                frame[p_new_y + dy][p_new_x + dx] = val

            # Prevent projection when location outside of image size 
            if not ((p_new_x >= c or p_new_y >= r) or (p_new_x < 0 or p_new_y < 0)):
                # Place value at location of projection in output image
                img_out[p_new_y][p_new_x] = val

    return img_out, frame

# Backward warping
def backward_warping(H, bounding_box, img_original, frame, dx, dy, nearest_neighbor = True):
    r, c = bounding_box
    y_max, x_max, _ = img_original.shape
    y_max -= 1 
    x_max -= 1

    y_frame, x_frame, _ = frame.shape
    y_frame -= 1
    x_frame -= 1

    H_inv = np.linalg.inv(H)

    for y_i in range(r[0], r[1]):
        for x_i in range(c[0], c[1]):
            # Get corresponding coordinates in original image
            _, p_new = rnsc.project_p(H_inv, x_i, y_i)

            p_new_x = p_new[0][0]
            p_new_y = p_new[1][0]

            if nearest_neighbor:
                p_new_x = round(p_new_x)
                p_new_y = round(p_new_y)

            if (p_new_x < 0 or p_new_x > x_max) or (p_new_y < 0 or p_new_y > y_max):
                continue 

            # Get original value
            val = img_original[p_new_y][p_new_x]        

            # Get corresponding coordinates in frame
            x_i_frame = x_i + dx
            y_i_frame = y_i + dy

            # Project on frame
            if not ((x_i_frame < 0 or x_i_frame > x_frame) or (y_i_frame < 0 or y_i_frame > y_frame)):
                # Set value in frame image
                frame[y_i + dy][x_i + dx] = val

    return frame

""" 
    Stitches images in sequential order given a set of images.
    Code has been implemented to work for a set of two images were the first image is warped towards the second.
    However can be easily extended such that each image gets warped to the correct other image in the set.
"""
def stitch_images(images, images_gray, img_names):
    kpm.images = images
    kpm.images_gray = images_gray 
    kpm.sift()
    kpm.bfm(visualize=True)
    
    img1 = images[0]
    img2 = images[1]

    H = rnsc.ransac(kpm.matches[0], kpm.keypoints[0], kpm.keypoints[1])

    # Compute size of stitched images frame
    shape, delta_y, delta_x = estimate_size_frame(H, img2)

    # Translation matrix M from position in img2 to position in the frame
    M = np.float32([[1, 0, delta_x],
                    [0, 1, delta_y]])

    # Choose img2 as reference image so use that the construct the frame
    frame = cv.warpAffine(img2, M, (shape[1], shape[0]))

    # Get bounding box of image to warp
    y_range, x_domain = get_bounding_box(H, img1)

    img1_on_img2, _ = forward_warping(H, img1, np.zeros(img1.shape), delta_x, delta_y)

    # Use backward warping to project image on frame
    img_proj = backward_warping(H, [y_range.astype(int), x_domain.astype(int)], img1, frame, delta_x, delta_y)

    # Convert color space back to RGB for display
    img_proj = img_proj[:, :, [2, 1, 0]]
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    img1_on_img2 = cv.cvtColor(img2[:, :, [2, 1, 0]], cv.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize = (20, 40))
    fig.suptitle("Projective transformation using an estimated homography through RANSAC")

    ax1 = fig.add_subplot(2, 2, 1)   
    plt.imshow(img1)
   
    ax2 = fig.add_subplot(2, 2, 2) 
    plt.imshow(img2)   
   
    ax3 = fig.add_subplot(2, 2, 3) 
    plt.imshow(img1_on_img2)   
   
    ax4 = fig.add_subplot(2, 2, 4)   
    plt.imshow(img_proj.astype(int))
   
    ax1.title.set_text(f"'{img_names[0]}'")
    ax2.title.set_text(f"'{img_names[1]}' chosen as reference image")
    ax3.title.set_text(f"Forward warping '{img_names[0]}' -> '{img_names[1]}'")
    ax4.title.set_text(f"Backward warping '{img_names[0]}' -> stich frame")
   
    plt.show()

if __name__ == "__main__":
    # Randomly choose default images to stitch
    if random.randint(0, 1):
        composition = ["left.jpg", "right.jpg"]
    else:
        composition = ["boat1.pgm", "boat2.pgm"]
        
    # composition = ["left.jpg", "right.jpg"]

    # Image data
    images = []
    images_gray = []
    
    # Ask user for manual input images
    default = uiux.single_yes_or_no_question("Stich default images together?")

    if default:
        print(f"We selected pictures '{composition[0]}' and '{composition[1]}' for you. Try again and we might pick another set \u2764")
    print(f"HERE WE GO!")
    uiux.print_animation()    
    print(f"We will stitch a set of {len(composition)} images.")

    if not default:
        print("Please provide us:")
        for i in range(len(composition)):
            composition[i] = uiux.get_string_input_stripped(f"Filename image {i + 1}: ")

    # Save images data
    for image in composition:
        img = read_image(image)

        # Force images in a color space that is not gray
        try:
            if len(img.shape) != 3:
                raise Exception("Sorry currently we only process colored images! Color images are required.")
        except Exception as error:
            uiux.print_error(f"{error}")
            exit()

        img_gray = BGR2GRAY(img)

        images.append(img)
        images_gray.append(img_gray)

    # Find the best transformation
    stitch_images(images, images_gray, composition)

    print(f"======================\u2605")
    print(f"We've stitched pictures '{composition[0]}' and '{composition[1]}' for you. Try again and we might pick another set \u2764")
    print(f"======================\u2605")