import numpy as np
import matplotlib.pyplot as plt
# import cv2

def estimate_alb_nrm( image_stack, scriptV, shadow_trick):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])

            
    for i in range(h):
        for j in range(w):
            vectori = image_stack[i,j]
            #print(type(vectori))
            #print(vectori.shape)
            scriptI = np.diag(vectori)
            #print(scriptI.shape)
            if shadow_trick:
                IV = np.dot(scriptI,scriptV)
                Ii = np.dot(scriptI,vectori)
                g = np.linalg.lstsq(IV,Ii,rcond=None)[0]
            else:
                g = np.linalg.lstsq(scriptV,vectori,rcond=None)[0]
            albedo[i,j] = np.linalg.norm(g)
            #print(albedo[i,j])
            normal[i,j] = (g / np.linalg.norm(g)).T
    print(albedo.shape,normal.shape)
    
    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point  
        scriptI * i pixel value
        albedo at this point is |g|
        normal at this point is g / |g|
    """
    fig = plt.figure()
    albedo_max = albedo.max()
    albedo_max = 1
    albedo = albedo / albedo_max
    print(albedo.shape)
    plt.imshow(albedo, cmap="gray")
    plt.show()
    
    # showing normals as three separate channels
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normal[..., 0])
    ax2 = figure.add_subplot(132)
    ax2.imshow(normal[..., 1])
    ax3 = figure.add_subplot(133)
    ax3.imshow(normal[..., 2])
    plt.show()
    
    return albedo, normal

    
if __name__ == '__main__':
     n = 5
     image_stack = np.zeros([10,10,n])
     scriptV = np.zeros([n,3])
     estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)
    
