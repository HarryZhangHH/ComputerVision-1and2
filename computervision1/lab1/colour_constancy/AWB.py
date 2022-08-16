import cv2
import numpy as np
import matplotlib.pyplot as plt


def grey_world(original):
    
    newImage = original.astype(np.float32)
    
    r= original[:,:,0]
    g= original[:,:,1]
    b= original[:,:,2]
    avgR = np.mean(r)
    avgG = np.mean(g)
    avgB = np.mean(b)
    greyValue = 128 
    newImage[:,:,0] = greyValue/avgR * r
    newImage[:,:,1] = greyValue/avgG * g
    newImage[:,:,2] = greyValue/avgB * b
    

    fig=plt.figure(num=1,figsize=(8,8))
    ax1=fig.add_subplot(221)
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2=fig.add_subplot(222)
    ax2.imshow(newImage/newImage.max())
    ax2.set_title("New image")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.savefig('grey_world')
    plt.show()

if __name__ == '__main__':
    original = cv2.imread('/home/zhanghh/lab1/colour_constancy/awb.jpg')[:, :, ::-1]

    grey_world(original)

