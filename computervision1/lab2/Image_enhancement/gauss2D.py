def gauss2D( sigma_x, sigma_y , kernel_size ):
    
    Gx = gauss1D(sigma_x, kernel_size)
    Gy = gauss1D(sigma_y, kernel_size)
    
    G = Gx.T * Gy
    return G
