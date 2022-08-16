import numpy as np
import cv2
import matplotlib.pyplot as plt

def denoise ( image , kernel_type , ** kwargs ):
    if kernel_type == 'box': 
        imOut = cv2.blur( image, ksize = kwargs['size'])
    
    elif kernel_type == 'median':
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
        imOut = cv2.medianBlur( image, ksize = kwargs['size'])
        imOut = cv2.cvtColor(imOut, cv2.COLOR_BGR2GRAY)
    
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur( image, ksize = kwargs['size'], sigmaX = kwargs['sigma'])
    
    else:
        print('Operation not implemented')
        
    return imOut



if __name__ == '__main__':
    image_saltpepper = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1_saltpepper.jpg').astype(np.float32)
    image_gaussian = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1_gaussian.jpg').astype(np.float32)
    image1 = plt. imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1.jpg').astype(np.float32)
    
    box_psnr = []
    median_psnr = []
    images = [0] * 4
    images[0] = image_saltpepper
    for i in range(3):
        boxing_size = [(3,3), (5,5), (7,7)]
        result = denoise(image_gaussian, kernel_type = 'box', size = boxing_size[i]).astype(np.float32)
        images[i+1] = result
        box_psnr.append(myPSNR(image1, result))
    titles = ['Origin image', 'Box filtering 3*3', 'Box filtering 5*5', 'Box filtering 7*7']
    fig=plt.figure(num=1,figsize=(14,14))
    for i in range(4):
        plt.subplot(1, 4, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.xticks([]), plt.yticks([])
#     plt.savefig('box_filtering_gaussian', bbox_inches='tight', pad_inches=0)
    plt.show()
    print(myPSNR(image1, image_saltpepper))
    print(box_psnr)
    
    
    for i in range(3):
        median_size = [3, 5, 7]
        result = denoise(image_gaussian, kernel_type = 'median', size = median_size[i]).astype(np.float32)
        images[i+1] = result
        median_psnr.append(myPSNR(image1, result))
    
    titles = ['Origin image', 'Median filtering 3*3', 'Median filtering 5*5', 'Median filtering 7*7']
    fig=plt.figure(num=1,figsize=(14,14))
    for i in range(4):
        plt.subplot(1, 4, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.xticks([]), plt.yticks([])
#     plt.savefig('median_filtering_gaussian', bbox_inches='tight', pad_inches=0)
    plt.show()
    print(median_psnr)
    
    
    gaussian_psnr = []
    images = [0] * 9
    image_gaussian = plt.imread('/home/zhanghh/桌面/lab2/code/Image_enhancement/images/image1_gaussian.jpg').astype(np.float32)
    for i in range(3):
        gaussian_size = [(3,3), (5,5), (7,7)]
        sigma = [0, 1, 2]
        for j in range(3):
            result = denoise(image_gaussian, kernel_type = 'gaussian', size = gaussian_size[i], sigma = sigma[j]).astype(np.float32)
            images[i*3 + j] = result
            gaussian_psnr.append(myPSNR(image1, result))
    
    titles = ['sigma 0 Gaussian filtering 3*3', 'sigma 0.5 Gaussian filtering 3*3', 'sigma 1 Gaussian filtering 3*3',
              'sigma 0 Gaussian filtering 5*5', 'sigma 0.5 Gaussian filtering 5*5', 'sigma 1 Gaussian filtering 5*5',
              'sigma 0 Gaussian filtering 7*7', 'sigma 0.5 Gaussian filtering 7*7', 'sigma 1 Gaussian filtering 7*7']
    fig=plt.figure(num=1,figsize=(14,14))
    fig.set_tight_layout(True)
    for i in range(9):
        plt.subplot(3, 3, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.xticks([]), plt.yticks([])
    plt.savefig('gaussian_filtering_saltpepper', bbox_inches='tight', pad_inches=0)
    plt.show()
    print(gaussian_psnr)
    
#     print(myPSNR(image_gaussian, denoise(image_gaussian, kernel_type = 'gaussian', size = (5,5), sigma = 0.5).astype(np.float32)))
#     print(myPSNR(image_gaussian, denoise(image_gaussian, kernel_type = 'gaussian', size = (7,7), sigma = 0.5).astype(np.float32)))

