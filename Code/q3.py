import numpy as np
import open3d as o3d
import os
import copy
from tqdm import tqdm
from scipy.spatial import KDTree
from em_icp import icp, transform

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

def clean_bg(pcd):
    pcd[pcd < -2] = np.nan
    pcd_arr_cleaned = pcd[~np.isnan(pcd).any(axis=1), :]
    return pcd_arr_cleaned

# def transform_pcd(source, R, t):
#     return R.dot(source) + t

def show(N=[1,2,4,10], iter=100, sample_func="random", method="kdtree", nr_samples=6000):
    for n in N:
        print(f'---N is {n}---')
        pcd_all = np.asarray([])
        last_pcd = np.asarray([])
        R_total, T_total = [], []
        for i in tqdm(range(0, iter, n)):
            if i < 10:
                pcd = o3d.io.read_point_cloud("Data/data/000000000"+str(i)+".pcd")
            else:
                pcd = o3d.io.read_point_cloud("Data/data/00000000"+str(i)+".pcd")
            
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # convert into ndarray
            pcd_arr = np.asarray(pcd.points)
            # ***  you need to clean the point cloud using a threshold ***
            pcd_arr = clean_bg(pcd_arr)

            if pcd_all.size == 0:
                pcd_all = pcd_arr
                last_pcd = pcd_arr
            else:
                R, T = icp(pcd_arr, last_pcd, sample_func, method, nr_samples)
                R_total.append(R)
                T_total.append(T)
                last_pcd = pcd_arr
                for ind, r in reversed(list(enumerate(R_total))):
                    pcd_arr = transform(pcd_arr, r, T_total[ind])
                pcd_all = np.concatenate((pcd_all, pcd_arr))
        # visualization from ndarray
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(pcd_all)
        o3d.visualization.draw_geometries([vis_pcd])
        o3d.io.write_point_cloud("N="+str(n)+"&"+sample_func+"&"+method+".pcd", vis_pcd)

#  Iteratively merge and estimate the camera poses for the consecutive frames.

def show_2(N=[4,10], iter=100, sample_func="random", method="kdtree", nr_samples=6000):
    for n in N:
        print(f'---N is {n}---')
        pcd_all = np.asarray([])
        for i in tqdm(range(0, iter, n)):
            if i < 10:
                pcd = o3d.io.read_point_cloud("Data/data/000000000"+str(i)+".pcd")

            else:
                pcd = o3d.io.read_point_cloud("Data/data/00000000"+str(i)+".pcd")
            
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # convert into ndarray
            pcd_arr = np.asarray(pcd.points)
            # ***  you need to clean the point cloud using a threshold ***
            pcd_arr = clean_bg(pcd_arr)

            if pcd_all.size == 0:
                pcd_all = pcd_arr
            else:
                R, T = icp(pcd_arr, pcd_all, sample_func, method, nr_samples)
                
                pcd_arr = transform(pcd_arr, R, T)
                pcd_all = np.concatenate((pcd_all, pcd_arr))
        # visualization from ndarray
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(pcd_all)
        o3d.visualization.draw_geometries([vis_pcd])
        o3d.io.write_point_cloud("N="+str(n)+"&"+sample_func+"&"+method+"_2.pcd", vis_pcd)

def main():
    # sample_func = "multires"
    # method = "kdtree"
    nr_samples = 6000
    q = input("Which question 1 or 2? ")
    sample_func = input("Which sample_function you what to use? (multires or random or uniform or inforeg) ")
    method = input("Which method you want to use? (kdtree or z-buffer) ")
    N = input("How many frame intervel? (1, 2, 4, 10 or all) ")
    if q == '2':
        print(q, N)
        if N in ['4','10']:
            if sample_func in ['multires' ,'random' ,'uniform' ,'inforeg'] and method in ['kdtree', 'z-buffer']:
                print(sample_func)
                show_2(N=[int(N)],  sample_func=sample_func, method=method, nr_samples=nr_samples)
            else:
                show_2(N=[int(N)])
        elif N == 'all':
            if sample_func in ['multires' ,'random' ,'uniform' ,'inforeg'] and method in ['kdtree', 'z-buffer']:
                show_2(sample_func=sample_func, method=method, nr_samples=nr_samples)
            else:
                show_2()
        else:
            show_2()
    else:
        if N in ['1','2','4','10']:
            if sample_func in ['multires' ,'random' ,'uniform' ,'inforeg'] and method in ['kdtree', 'z-buffer']:
                show(N=[int(N)],  sample_func=sample_func, method=method, nr_samples=nr_samples)
            else:
                show(N=[int(N)])
        elif N == "all":
            if sample_func in ['multires' ,'random' ,'uniform' ,'inforeg'] and method in ['kdtree', 'z-buffer']:
                show(sample_func=sample_func, method=method, nr_samples=nr_samples)
            else:
                show()
        else:
            show()

if __name__ == "__main__":
    main()