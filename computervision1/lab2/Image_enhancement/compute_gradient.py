import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal

def compute_gradient ( image ):
    
#     if (gradient == "x"):
    x = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
    Gx = scipy.signal.convolve2d( image, x)
        
#     if (gradient == "y"):
    y = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
    Gy = scipy.signal.convolve2d( image, y) 
        
#     if (gradient == "magnitude"):
    im_mag = np.sqrt(np.square(Gx) + np.square(Gy))
        
#     if (gradient == "direction"):
    im_dir = np.power(np.tan(Gy/Gx), -1) //?????????
    
    im_dir2 = np.arctan(Gy/Gx) //?????
    
    return Gx , Gy , im_mag , im_dir2
    
if __name__ == '__main__':
    original_image = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image2.jpg').astype(np.float32)
    
    Gx , Gy , im_mag , im_dir = compute_gradient(original_image)
    
    images = []
    images.append(Gx)
    images.append(Gy)
    images.append(im_mag)
    images.append(im_dir)
    
    titles = ['Gx', 'Gy', 'magnitude', 'direction']
    fig=plt.figure(num=1,figsize=(14,14))
    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i], fontsize=20)
        plt.xticks([]), plt.yticks([])
#     plt.savefig('compute_gradient')
    plt.show()

