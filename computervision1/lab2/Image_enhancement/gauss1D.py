import numpy as np

def gauss1D( sigma , kernel_size )
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
	
    size = np.floor(ksize/2)
    for index,x in enumerate(np.arange(-size, size+1)):
        G[0][index] = (1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((x**2)/(2*sigma**2))))
    Gaussian_sum = np.sum(G)
    G = G/Gaussian_sum
	
	return G
