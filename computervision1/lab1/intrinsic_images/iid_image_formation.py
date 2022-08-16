import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_reconstruction(albedo, shading, original):
    
    reconstruction = np.multiply(albedo/255,shading/255)
    fig=plt.figure(num=1,figsize=(8,8))
    ax1=fig.add_subplot(221)
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2=fig.add_subplot(222)
    ax2.imshow(reconstruction)
    ax2.set_title("Reconstruction")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3=fig.add_subplot(223)
    ax3.imshow(albedo)
    ax3.set_title("Albedo")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4=fig.add_subplot(224)
    ax4.imshow(shading)
    ax4.set_title("Shading")
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.show()

if __name__ == '__main__':
    original = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball.png')[:, :, ::-1]
    albedo = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball_albedo.png')[:, :, ::-1]
    shading = cv2.imread('/home/zhanghh/lab1/intrinsic_images/ball_shading.png')

    image_reconstruction(albedo, shading, original)
