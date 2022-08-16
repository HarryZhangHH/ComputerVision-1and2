import numpy as np
import cv2
import os
from utils import *
# from estimate_alb_nrm import estimate_alb_nrm
# from check_integrability import check_integrability
# from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='/home/zhanghh/lab1/photometric/photometrics_images/MonkeyGray', channel=0 ):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    [image_stack, scriptV] = load_syn_images(image_dir, channel)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print(image_stack)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=True)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q, 'column' )
    
    height_map_2 = construct_surface( p, q, 'row' )
    
    height_map_3 = construct_surface( p, q, 'average' )
    # show results
    show_results(albedo, normals, height_map, SE)

def photometric_stereo_color_channel(image_dir='/home/zhanghh/lab1/photometric/photometrics_images/MonkeyColor/', channel=3):
#     for i in range(channel):
    photometric_stereo(image_dir, 1)
    
## Face
def photometric_stereo_face(image_dir='/home/zhanghh/lab1/photometric/photometrics_images/yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, True)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q , 'row')

    # show results
    show_results(albedo, normals, height_map, SE)
    
if __name__ == '__main__':
#     photometric_stereo()
    photometric_stereo_face()
#     photometric_stereo_color_channel()
