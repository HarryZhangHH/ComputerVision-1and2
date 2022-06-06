import sys
import os
# import dlib
import glob
import numpy as np
import tqdm as tqdm


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords


def detect_landmark(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # Draw the face landmarks on the screen.
        return shape_to_np(shape)



def save_obj(file_path, shape, color, triangles):
    assert len(shape.shape) == 2
    assert len(color.shape) == 2
    assert len(triangles.shape) == 2
    assert shape.shape[1] == color.shape[1] == triangles.shape[1] == 3
    assert np.min(triangles) == 0
    assert np.max(triangles) < shape.shape[0]

    with open(file_path, 'wb') as f:
        data = np.hstack((shape, color))
        
        np.savetxt(
            f, data,
            fmt=' '.join(['v'] + ['%.5f'] * data.shape[1]))

        np.savetxt(f, triangles + 1, fmt='f %d %d %d')


def render(uvz, color, triangles, H=480, W=640):
    """ Renders an image of size WxH given u, v, z vertex coordinates, vertex color and triangle topology.
    
    uvz - matrix of shape Nx3, where N is an amount of vertices
    color - matrix of shape Nx3, where N is an amount of vertices, 3 channels represent R,G,B color scaled from 0 to 1
    triangles - matrix of shape Mx3, where M is an amount of triangles, each column represents a vertex index
    """

    assert len(uvz.shape) == 2
    assert len(color.shape) == 2
    assert len(triangles.shape) == 2
    assert uvz.shape[1] == color.shape[1] == triangles.shape[1] == 3
    assert np.min(triangles) == 0
    assert np.max(triangles) < uvz.shape[0]

    def bbox(v0, v1, v2):
        u_min = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        u_max = int(min(W - 1, np.ceil(max(v0[0], v1[0], v2[0]))))

        v_min = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        v_max = int(min(H - 1, np.ceil(max(v0[1], v1[1], v2[1]))))

        return u_min, u_max, v_min, v_max

    def cross_product(a, b):
        return a[0] * b[1] - b[0] * a[1]

    p = np.zeros([3])

    z_buffer = -np.ones([H, W]) * 100500

    image = np.zeros([H, W, 3])

    for triangle in tqdm.tqdm(triangles):
        id0, id1, id2 = triangle
        v0 = uvz[int(id0)]
        v1 = uvz[int(id1)]
        v2 = uvz[int(id2)]
        v02 = v2 - v0
        v01 = v1 - v0

        u_min, u_max, v_min, v_max = bbox(v0, v1, v2)

        # double triangle signed area
        tri_a = cross_product(v1 - v0, v2 - v0)
        for v in range(v_min, v_max + 1):
            p[1] = v
            for u in range(u_min, u_max + 1):

                p[0] = u
                v0p = p - v0

                b1 = cross_product(v0p, v02) / tri_a
                b2 = cross_product(v01, v0p) / tri_a

                if (b1 < 0) or (b2 < 0) or (b1 + b2 > 1):
                    continue
                
                b0 = 1 - b1 - b2
                p[2] = b0 * v0[2] + b1 * v1[2] + b2 * v2[2]


                if p[2] > z_buffer[v, u]:
                    z_buffer[v, u] = p[2]
                    image[v, u] = b0 * color[int(id0)] + b1 * color[int(id1)] + b2 * color[int(id2)]
    
    return image
