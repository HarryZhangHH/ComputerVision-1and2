import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def myPSNR(orig_image, approx_image):
    
    difference = orig_image - approx_image
    mse = np.mean(np.square(difference))
    psnr = 20 * np.log10(orig_image.max()/np.sqrt(mse))
    
    psnr2 = peak_signal_noise_ratio(orig_image, approx_image, data_range=orig_image.max())
    
    return psnr

if __name__ == '__main__':
    orig_image = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1_saltpepper.jpg').astype(np.float32)
    approx_image = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1.jpg').astype(np.float32)
    print(myPSNR(orig_image, approx_image))
