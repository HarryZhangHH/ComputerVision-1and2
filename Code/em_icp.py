import numpy as np
import open3d as o3d
import os
import copy
from tqdm import tqdm
from scipy.spatial import KDTree
from itertools import product

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.


######                                                           ######
##      notice: This is just some example, feel free to adapt        ##
######                                                           ######


# == Load data ==
def open3d_example():
    pcd = o3d.io.read_point_cloud("Data/data/0000000000.pcd")
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    pcd_arr_cleaned = pcd_arr

    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
    o3d.visualization.draw_geometries([vis_pcd])


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target


############################
#     ICP                  #
############################
###### 0. (adding noise)


###### 1. initialize R= I , t= 0
def initialize(a1):
    return np.identity(a1.shape[1]), np.zeros(a1.shape[1]).T

###### go to 2. unless RMS is unchanged(<= epsilon)
def converged(rms, prev_rms, epsilon=3e-6):
    if abs(rms - prev_rms) <= epsilon:
        return True
    return False

###### 2. using different sampling methods
def sample(source, target, nr_samples, sample_func):
    max_samples = np.min((len(source), len(target)))
    if nr_samples > max_samples:
        nr_samples = max_samples
    if sample_func == "uniform":
        samples_index_target = np.random.randint(len(target), size=nr_samples)
        samples_index_source = np.random.randint(len(source), size=nr_samples)
        return source[samples_index_source], target[samples_index_target]
    if sample_func == "random" or sample_func == "multires":
        samples_index_target = np.random.choice(len(target), nr_samples, replace=False)
        samples_index_source = np.random.choice(len(source), nr_samples, replace=False)
        return source[samples_index_source], target[samples_index_target]
    elif sample_func == "inforeg":
        src_pcd = o3d.geometry.PointCloud()
        tar_pcd = o3d.geometry.PointCloud()

        src_pcd.points = o3d.utility.Vector3dVector(source)
        tar_pcd.points = o3d.utility.Vector3dVector(target)

        source_kps = o3d.geometry.keypoint.compute_iss_keypoints(src_pcd)
        target_kps = o3d.geometry.keypoint.compute_iss_keypoints(tar_pcd)

        source = np.asarray(source_kps.points)
        target = np.asarray(target_kps.points)

        if len(source) > len(target):
            return source[np.random.randint(len(source), size=len(target)), :], target
        else:
            return source, target[np.random.randint(len(target), size=len(source)), :]


###### 3. transform point cloud with R and t
def transform(a1, R, t):
    a_1 = np.ones((a1.shape[1]+1, a1.shape[0]))
    a_1[0:3, :] = a1.T
    # # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    new_a1 = np.dot(T, a_1)
    return new_a1[0:3, :].T

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach
def closest_points(a1, a2):
    a1_size = a1.shape[0]
    closest_points = []
    distances = []
    for i in range(a1_size):
        a1_point = a1[i]
        distances = list(np.sum((a2 - a1_point) ** 2, axis=1))
        min_dist = min(distances)
        cp_index = distances.index(min_dist)
        distances.append(min_dist)
        closest_points.append(cp_index)
    return closest_points, np.array(distances)

###### 5. Calculate RMS
def RMS_calc(distances):
    return np.sqrt(np.mean(distances))

###### 6. Refine R and t using SVD
def svd(a1, a2):
    # find center
    # translate points to their centroids
    center_a1 = np.mean(a1, axis=0)
    center_a2 = np.mean(a2, axis=0)
    a1_centered = a1 - center_a1
    a2_centered = a2 - center_a2

    # rotation matrix
    H = np.dot(a1_centered.T, a2_centered)
    U, S, V_T = np.linalg.svd(H)
    R = np.dot(V_T.T, U.T)

    # translation
    t = center_a2.T - np.dot(R, center_a1.T)

    return R, t

###### 7. Kd-Tree
def kd_tree(source, target):
    tree = KDTree(np.c_[target])
    indices, distances = [], []
    for i in source:
        dd, ii = tree.query(i, k=1)
        distances.append(dd)
        indices.append(ii)

    return indices, distances

###### 8 Z-Buffer
def eucliDist(A,B):
    # calculate the distance of two points
    return np.sqrt(sum(np.power((A - B), 2)))
def z_buffer(source, target, H=20, W=20, m=5):
    xx = [min(min(target.T[0]), min(source.T[0])), max(max(target.T[0]), max(source.T[0]))]
    yy = [min(min(target.T[1]), min(source.T[1])), max(max(target.T[1]), max(source.T[1]))]

    # build the projection area
    x = np.linspace(xx[0], xx[1], H)
    y = np.linspace(yy[0], yy[1], W)
    # X, Y = np.meshgrid(x, y)
    z_buffer_A1=np.zeros((H,W))
    # suppose each pixel in projection plane is d(i,j) = np.inf
    z_buffer_A1+=np.inf
    location_A1 = np.zeros((H,W,2))
    location_A1+=np.nan
    z_buffer_A2=np.zeros((H,W))
    # suppose each pixel in projection plane is d(i,j) = np.inf
    z_buffer_A2+=np.inf
    location_A2 = np.zeros((H,W,2))
    location_A2+=np.nan

    dist_idx = np.zeros((H,W))
    # print('Project points of source into the source buffer, using the nearest point and record the geometry information')
    for x_s,y_s,z_s in source.T[:]:
        for i, j in product(range(H), range(W)):
            dist_idx[i,j] = eucliDist(np.array([x_s,y_s]), np.array([x[i], y[j]]))
        x_d, y_d = np.where(dist_idx==np.min(dist_idx))
        if z_s < z_buffer_A1[x_d,y_d]:
            z_buffer_A1[x_d,y_d] = z_s
            location_A1[x_d,y_d,0] = x_s
            location_A1[x_d,y_d,1] = y_s

    dist_idx = np.zeros((H,W))
    # print('Project points of target into the target buffer, using the nearest point and record the geometry information')
    for x_t,y_t,z_t in target.T[:]:
        for i, j in product(range(H), range(W)):
            dist_idx[i,j] = eucliDist(np.array([x_t,y_t]), np.array([x[i], y[j]]))
        x_d, y_d = np.where(dist_idx==np.min(dist_idx))
        if z_t < z_buffer_A2[x_d,y_d]:
            z_buffer_A2[x_d,y_d] = z_t
            location_A2[x_d,y_d,0] = x_t
            location_A2[x_d,y_d,1] = y_t
    # the last step map each point in source buffer with the target point in target buffer in m*m window using nearest distance
    distances = []
    new_source = np.array([])
    new_target = np.array([])
    dist_idx = np.zeros((H,W))
    dist_idx += np.inf
    for h, w in product(range(H), range(W)):
        if np.isnan(location_A1[h,w][0]):
            continue
        for h2, w2 in product(range(h-round(m/2), h+round(m/2)), range(w-round(m/2), w+round(m/2))):
            if h2<0 or h2>=H or w2<0 or w2>=W or np.isnan(location_A2[h2,w2][0]):
                continue
            else:
                dist_idx[h2,w2] = eucliDist(np.array(location_A1[h,w]), np.array(location_A2[h2,w2]))
        if np.min(dist_idx) != np.inf:
            min_dist = np.min(dist_idx)
            x_t, y_t = np.where(dist_idx==min_dist)
            distances.append(min_dist)
            if new_source.size == 0:
                new_source = np.array([location_A1[h,w][0], location_A1[h,w][1], z_buffer_A1[h,w]]).reshape(3,1)
                new_target = np.array([location_A2[int(x_t),int(y_t)][0], location_A2[int(x_t),int(y_t)][1], z_buffer_A2[int(x_t),int(y_t)]]).reshape(3,1)
            else:
                new_source = np.hstack((new_source, np.array([location_A1[h,w][0], location_A1[h,w][1], z_buffer_A1[h,w]]).reshape(3,1)))
                new_target = np.hstack((new_target, np.array([location_A2[int(x_t),int(y_t)][0], location_A2[int(x_t),int(y_t)][1], z_buffer_A2[int(x_t),int(y_t)]]).reshape(3,1)))
    return new_source, new_target, distances


def icp(source, target, sample_func,  method, nr_samples=None):
    n_points = 100
    if sample_func == "uniform" or sample_func == "random":
        sampled_source, sampled_target = sample(source, target, nr_samples, sample_func)
    elif sample_func == 'multires':
        sampled_source, sampled_target = sample(source, target, n_points, sample_func)
    elif sample_func == 'inforeg':
        sampled_source, sampled_target = sample(source, target, nr_samples, sample_func)
    else:
        sampled_source, sampled_target = source, target

    src = copy.deepcopy(sampled_source)
    tar = copy.deepcopy(sampled_target)

    N = 5
    max_iterations = 40
    prev_error = np.inf
    max_samples = min([len(source), len(target)])

    R_total, t_total = [], [] 

    for i in (range(max_iterations)):
        # find the nearest neighbours between the current source and destination points
        if method == 'zbuffer':
            src_x, tar_x, distances = z_buffer(src.T, tar.T)
            try:
                R, t = svd(src_x.T, tar_x.T)
            except:
                R, t = svd(sampled_source, src)
                return R, t
            # check error
            error = RMS_calc(distances)
        else:
            if method == 'kdtree':
                indices, distances = kd_tree(src, tar)
            # if method == 'zbuffer':
            #     indices, distances = z_buffer(src, tar)
            else:
                indices, distances = closest_points(src, tar)

            # check error
            error = RMS_calc(distances)

            if converged(error, prev_error) and sample_func != "multires":
                R, t = svd(sampled_source, src)
                return R, t

            R, t = svd(src, tar[indices])
        R_total.append(R)  
        t_total.append(t)  
        # compute the transformation between the current source and nearest destination points
        if sample_func == "random":
            sampled_source, sampled_target = sample(source, target, nr_samples, sample_func)
            src = copy.deepcopy(sampled_source)
            tar = copy.deepcopy(sampled_target)
            for ind, r in reversed(list(enumerate(R_total))): 
                src = transform(src, r, t_total[ind])  
        elif sample_func == "multires" and np.abs(prev_error - error) < 1e-3 and N > 2:
            N -= 1
            nr_samples = int(max_samples / N)
            sampled_source, sampled_target = sample(source, target, nr_samples, sample_func)
            src = copy.deepcopy(sampled_source)
            tar = copy.deepcopy(sampled_target)

            for ind, r in reversed(list(enumerate(R_total))):  
                src = transform(src, r, t_total[ind])  
        else:
            # update the current source
            src = transform(src, R, t)
        
        prev_error = error
    R, t = svd(sampled_source, src)
    return R, t


def main():
    source, target = open_bunny_data()
    sample_func = "multires"
    method = "kdtree"
    nr_samples = 6000
    R, t = icp(source.T, target.T, sample_func, method, nr_samples)
    transformed_source = transform(source.T, R, t)
    vis_pcd = o3d.geometry.PointCloud()
    vis1_pcd = o3d.geometry.PointCloud()
    vis2_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(source.T)
    vis1_pcd.points = o3d.utility.Vector3dVector(transformed_source)
    vis2_pcd.points = o3d.utility.Vector3dVector(target.T)
    vis_pcd.paint_uniform_color([1, 0, 1])
    vis1_pcd.paint_uniform_color([0, 0, 1])
    vis2_pcd.paint_uniform_color([1, 1, 0])
    o3d.visualization.draw_geometries([vis_pcd, vis1_pcd, vis2_pcd])

if __name__ == "__main__":
    main()


############################
#  Additional Improvements #
############################