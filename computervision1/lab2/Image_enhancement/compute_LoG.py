import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal

def gaussian_filter (sigma, ksize):
    
    Gaussian1D = np.zeros((1, ksize), np.float32)
    size = np.floor(ksize/2)
    for index,x in enumerate(np.arange(-size, size+1)):
        Gaussian1D[0][index] = (1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((x**2)/(2*sigma**2))))
    Gaussian_sum = np.sum(Gaussian1D)
    Gaussian1D = Gaussian1D/Gaussian_sum
    Gaussian2D = Gaussian1D.T * Gaussian1D
    
    return Gaussian1D
    
    
def compute_LoG ( image , LOG_type, sigma, ksize ):

    if (LOG_type == 1):
        G = gaussian_filter( ksize = (5,5), sigmaX = 0.5)
        imOut = signal.convolve2d(image, G)
        imOut = cv2.Laplacian(imOut, cv2.CV_16S, ksize = 5)
        imOut = cv2.convertScaleAbs(imOut)
        
    if (LOG_type == 2):
#         LoG Kernel
        G = np.zeros((ksize, ksize), dtype=np.float)
        size = np.floor(ksize/2)
        for x in range(-size, size+1):
            for y in range(-size, size+1):
                G[y, x] = (x ** 2 + y ** 2 - sigma ** 2) * (1/(np.exp( (x ** 2 + y ** 2) / (2 * sigma ** 2))))
        G /= (2 * np.pi * (sigma ** 6))
        G /= sum(G)
        imOut = scipy.signal.convolve2d(image, G)
        
    if (LOG_type == 3):
        DoG1 = gaussian_filter(sigma1, ksize)
        DoG1 = scipy.signal.convolve2d(image, DoG1)
        DoG2 = gaussian_filter(sigma2, ksize)
        DoG2 = scipy.signal.convolve2d(image, DoG2)
        imOut = DoG1 - DoG2
#         imOut = diff/2
        
    
    return imOut

if __name__ == '__main__':
    G = gaussian_filter(2,5)
    print(G)
