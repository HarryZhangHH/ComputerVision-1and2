import numpy as np

def construct_surface(p, q, path_type):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    height_map[i, j] = 0
                elif j == 0:
                    height_map[i, 0] = height_map[i-1,0] + q[i,0]
                else:
                    height_map[i, j] = height_map[i, j-1] + p[i, j]
                    
    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    height_map[i, j] = 0
                elif i == 0:
                    height_map[0, j] = height_map[0,j-1] + q[0,j]
                else:
                    height_map[i, j] = height_map[i-1, j] + p[i, j]
    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        
        height_map_1 = construct_surface(p, q, 'column')
        height_map_2 = construct_surface(p, q, 'row')
        height_map = np.divide(height_map_1 + height_map_2, 2)
        
    return height_map
