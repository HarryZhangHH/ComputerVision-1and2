import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_recolor(albedo, shading, original):
            
    x,y,c = albedo.shape
    RGB = np.unique(albedo)
    print(RGB)
    albedo[:,:,0] = 0
    albedo[:,:,1] = np.ones([x,y])*255
    albedo[:,:,2] = 0
    
    recolor = np.multiply(albedo/255,shading/255)
    fig=plt.figure(num=1,figsize=(8,8))
    ax1=fig.add_subplot(221)
    ax1.imshow(original)
    ax1.set_title("original")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2=fig.add_subplot(222)
    ax2.imshow(recolor)
    ax2.set_title("recolor")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()

if __name__ == '__main__':
    original = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball.png')[:, :, ::-1]
    albedo = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball_albedo.png')[:, :, ::-1]
    shading = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball_shading.png')
    
    image_recolor(albedo, shading, original)
