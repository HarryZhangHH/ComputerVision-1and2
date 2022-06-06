import numpy as np
import random
import cv2
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import KDTree

DATA_DIR = './Data/House/'  # This depends on where this file is located. Change for your needs.

def ICP(base, target):
    center_target = np.mean(target, axis=1)
    center_sample = np.mean(base, axis=1)
    R = np.eye(3)
    t = np.array([0,0,0])
    
    KD_index = KDTree(target.T)
    last_distance = np.array([1,1,1])
    
    sample = copy.deepcopy(base)
    for i in range(200):
        transformed_sample = np.dot(sample.T, R) + t

        I = KD_index.query(transformed_sample)
        idx = I[1]
        
        distance = np.sqrt(np.sum(I[0])**2 / transformed_sample.shape[0])
        
        if np.sum(np.abs(last_distance - distance)) < 1e-10:
            break

        last_distance = distance
        
        enter_transformed_sample = transformed_sample.mean(axis=0)
        center_target = target[:, idx].T.mean(axis = 0)
        
        #Normalize
        A = np.dot( (transformed_sample - enter_transformed_sample).T, (target[:, idx].T - center_target) )
        
        U, s, V_t = np.linalg.svd(A)
        R = np.dot(U,V_t)
        t = center_target - np.dot(center_sample, R)
    
    return R, t, distance

def match_keypoints(image_1, image_2, show_image=False):
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)

    if show_image:
        matches_sample = random.sample(matches, 10)
        image_match = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches_sample, image_2, flags=2)
        plt.imshow(image_match)
        plt.axis('off')
        plt.show()

    return matches, [keypoints_1, descriptors_1], [keypoints_2, descriptors_2]

def fund_matrix(kp1, kp2):
    n, m = np.array(kp1).shape
    A = []
    for i in range(n):
        row = [kp1[i][0]*kp2[i][0], kp1[i][0]*kp2[i][1], kp1[i][0], kp1[i][1]*kp2[i][0], kp1[i][1]*kp2[i][1],
               kp1[i][1], kp2[i][0], kp2[i][1], 1]
        A = np.append(A, row)

    A = np.reshape(A, (n, 9))
    _, sigma, V_T = np.linalg.svd(A, full_matrices=True)

    F_vec = V_T.transpose()[:, np.argmin(sigma)]
    F = np.reshape(F_vec, (3, 3))

    FU, FD, FV_T = np.linalg.svd(F)
    FD_t = FU @ np.diag([*FD[:2], 0]) @ FV_T

    return FD_t

def norm_kps(kps):

    n, m = kps.shape
    ones = np.ones((n, 1))
    kps = np.hstack((kps, ones)).T

    m_x = np.mean(kps[0])
    m_y = np.mean(kps[1])

    d = np.mean(np.sqrt((kps[0]-m_x)**2 + (kps[1]-m_y)**2))
    T = np.identity(3)
    norm = np.sqrt(2/d)

    T[0, 0], T[1, 1] = norm, norm
    T[0, 2], T[1, 2] = -m_x*norm, -m_y*norm
    normed_kps = T @ kps

    return normed_kps.T, T


def ransac(kp1, kp2, iter=1000):
    inliers, best_inliers = 0, 0
    best_F, best_kp1, best_kp2 = 0, 0, 0
    nkp1, _ = norm_kps(kp1)
    nkp2, _ = norm_kps(kp2)
    best_T1, best_T2 = 0, 0

    for i in (range(iter)):

        idxs = list(np.arange(len(kp1)))
        eight_matches = random.sample(idxs, 8)
        eight_kp1 = np.array([kp1[idx] for idx in eight_matches])
        eight_kp2 = np.array([kp2[idx] for idx in eight_matches])

        eight_nkp1, T1 = norm_kps(eight_kp1)
        eight_nkp2, T2 = norm_kps(eight_kp2)
        F = fund_matrix(eight_nkp1, eight_nkp2) 


        for i in range(len(kp1)):
            p1 = nkp1[i]
            p2 = nkp2[i]
            top = (p1.T @ F @ p2)**2
            Fp1x = np.dot(F, p1)[0]**2
            Fp1y = np.dot(F, p1)[1]**2
            Fp2x = np.dot(F, p2)[0]**2
            Fp2y = np.dot(F, p2)[1]**2
            bottom = Fp1x + Fp1y + Fp2x + Fp2y
            d_i = top / bottom

            if d_i <= 0.00001:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_F = F
            best_kp1 = eight_kp1
            best_kp2 = eight_kp2
            best_T1 = T1
            best_T2 = T2

        inliers = 0

    best_F = best_T2.T @ best_F @ best_T1
    return best_kp1, best_kp2, best_F

def drawlines(img1,img2,lines,pts1,pts2):

    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(int(0),int(255),int(3)).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (int(x0),int(y0)), (int(x1),int(y1)), color, 1)
        img1 = cv2.circle(img1, tuple((int(pt1[0]), int(pt1[1]))), 5, color, -1)
        img2 = cv2.circle(img1, tuple((int(pt2[0]), int(pt2[1]))), 5, color, -1)

    return img1,img2

def sample(kp1, kp2):
    idxs = list(np.arange(len(kp1)))
    eight_matches = random.sample(idxs, 8)

    eight_kp1 = np.array([kp1[idx] for idx in eight_matches])
    eight_kp2 = np.array([kp2[idx] for idx in eight_matches])

    return eight_kp1, eight_kp2

def plot(img1, img2 ,kp1, kp2, F):
    lines1 = cv2.computeCorrespondEpilines(kp2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img2, img1 ,lines1, kp1, kp2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(kp1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2, img1,lines2, kp2, kp1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

    return 

def find_best_kp(left, right, method='nothing'):
    matches, kp_data1, kp_data2 = match_keypoints(left, right, show_image=False)
    kp1 = np.array([kp_data1[0][match.queryIdx].pt for match in matches])
    kp2 = np.array([kp_data2[0][match.trainIdx].pt for match in matches])

    if method == 'nothing':
        return kp1, kp2, np.array([])
    elif method == 'no normalized':
        eight_kp1, eight_kp2 = sample(kp1, kp2)
        F_them, mask = cv2.findFundamentalMat(eight_kp1, eight_kp2,cv2.FM_LMEDS)
        F = fund_matrix(eight_kp1, eight_kp2)
        return eight_kp1, eight_kp2, F
    else:
        nkp1, T1 = norm_kps(kp1)
        nkp2, T2 = norm_kps(kp2)
        eight_nkp1, eight_nkp2 = sample(nkp1, nkp2)
        if method == 'normalized':
            F_norm = fund_matrix(eight_nkp1, eight_nkp2)
            F_denorm = T2.T @ F_norm @ T1
            return eight_nkp1, eight_nkp2, F_norm
        elif method == 'ransac':
            best_kp1, best_kp2, F_rans = ransac(kp1, kp2, 100)
            return best_kp1, best_kp2, F_rans

def eucliDist(A,B):
    # calculate the distance of two points
    return np.sqrt(sum(np.power((A - B), 2)))

def chaining(decimal=0, method='nothing', threshold=3):
    pvm = np.array([[]]*(49*2))
    for i in tqdm(range(1,50)):
        if i < 9:
            left = cv2.imread(DATA_DIR + 'frame0000000'+str(i)+'.png', -1)
            right = cv2.imread(DATA_DIR + 'frame0000000'+str(i+1)+'.png', -1)
        elif i == 9:
            left = cv2.imread(DATA_DIR + 'frame0000000'+str(i)+'.png', -1)
            right = cv2.imread(DATA_DIR + 'frame000000'+str(i+1)+'.png', -1)
        elif i > 9 and i < 49:
            left = cv2.imread(DATA_DIR + 'frame000000'+str(i)+'.png', -1)
            right = cv2.imread(DATA_DIR + 'frame000000'+str(i+1)+'.png', -1)
        elif i == 49:
            left = cv2.imread(DATA_DIR + 'frame000000'+str(i)+'.png', -1)
            right = cv2.imread(DATA_DIR + 'frame0000000'+str(1)+'.png', -1)

        kp1, kp2, _ = find_best_kp(left, right, method)
        point, point2 = np.round(kp1, decimals=decimal), np.round(kp2, decimals=decimal)
        for j in range(len(point)):
            x = point[j][0]
            y = point[j][1]
            x2 = point2[j][0]
            y2 = point2[j][1]
            if eucliDist(point[j], point2[j]) > threshold:
                continue
            isin = False
            mask = np.isin(pvm[2*(i-1)], x)
            idx = np.argwhere(mask==True)
            for index in idx:
                if pvm[2*(i-1)+1][index] == y:
                    isin = True
                    break
            if i != 49:
                if isin:
                    pvm[2*(i-1)+2][index] = x2
                    pvm[2*(i-1)+3][index] = y2
                else:
                    new_pvm = np.zeros((49*2,1))
                    new_pvm[2*(i-1)] = x
                    new_pvm[2*(i-1)+1] = y 
                    new_pvm[2*(i-1)+2] = x2
                    new_pvm[2*(i-1)+3] = y2
                    pvm = np.hstack((pvm,new_pvm))
            else:
                isin2 = False
                mask2 = np.isin(pvm[0], x2)
                idx2 = np.argwhere(mask2==True)
                for index2 in idx2:
                    if pvm[1][index2] == y2:
                        isin2 = True
                        break
                if isin:
                    pvm[0][index] = x2
                    pvm[1][index] = y2
                elif isin2:
                    pvm[2*(i-1)][index2] = x
                    pvm[2*(i-1)+1][index2] = y
                else:
                    new_pvm = np.zeros((49*2,1))
                    new_pvm[2*(i-1)] = x
                    new_pvm[2*(i-1)+1] = y
                    new_pvm[0] = x2 
                    new_pvm[1] = y2 
                    pvm = np.hstack((pvm,new_pvm))
    return pvm

# ALGORITHM 
#    1. represent the input as a 2F x P measurement matrix W 
#    2. Compute SVD of W = USV'
#    3. Define M' = U_3(S_3)^(1/2), S' = (S_3)^(1/2)V'_3 (U_3 means the
#    first 3 x 3 block, where M' and S' are liner transformations of
#    the actual M and S
#    4. Compute Q(or C) by imposing the metric constraints i.e. let L = QQ' 
#    and solve AL = b for L, use cholseky to recover Q
#    5. Compute M and S using M', S', and Q

def data_centering(pvm):
    pvm = np.where(pvm,pvm,np.nan)   
    return (pvm.T - np.nanmean(pvm, axis=1)).T

def get_dense_submatrix(pvm, N, sampling):
    column = pvm[:,0]
    mask = ~np.isnan(column)
    valid_row = pvm[mask, :]

    if valid_row.shape[0] < N*2:
        return np.array([]), pvm[:,1:]

    overlapping_columns = (~np.isnan(valid_row[:N*2])).all(axis=0)

    #Change PVM to not overlap and dense submatrix to do overlap

    if sampling and valid_row.shape[0] >= 20:    
        overlapping_columns = (~np.isnan(valid_row[:10*2])).all(axis=0)
        sampling_idx = sorted(random.sample(range(1, 10) , (N-1)))
        denses = valid_row[:2, overlapping_columns]
        for i in sampling_idx:
            dense = valid_row[2*i:2*i+2, overlapping_columns]
            denses = np.vstack((denses, dense))
        pvm = pvm[:, ~overlapping_columns]
        dense = denses
    else:
        pvm = pvm[:, ~overlapping_columns]
        dense = valid_row[:N*2, overlapping_columns]
    return dense, pvm

def factorization(D, ambiguity, other_decom=False):
    D = data_centering(D)
    U, W, V_t = np.linalg.svd(D)
    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    V3_t = V_t[:3]

    if other_decom:
        motion = U3@W3
        structure = W3@V3_t
    else:
        motion = U3@np.sqrt(W3)   
        structure = np.sqrt(W3)@V3_t

    if ambiguity:
        structure = remove_affine_ambiguity(motion, structure)
    return structure

def plot_structure(structure):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(structure[0], np.negative(structure[1]), np.negative(structure[2]), s=7, marker='.')
    # for ii in range(0,360,90):
    #     for jj in range(10,100,40):
    #         ax.view_init(elev=jj, azim=ii)
    #         plt.savefig("pvm_result/movie%d_%d.png" % (ii, jj))
    plt.show()

def improve_density(pvm, N):
    mask = (~np.isnan(pvm)).sum(0) >= N
    return pvm[:, mask]

def remove_affine_ambiguity(motion, structure):
    # Compute C by imposing the metric constraints i.e. let L = CC' and solve Gl = c for l, use cholseky to recover C
       
    m = int(motion.shape[0]/2)
    A = motion
    b = np.zeros((m * 2,3))
    for i in range(m):
        b[(i*2) : (i*2)+2, :] = np.linalg.pinv(A[(i*2) : (i*2)+2,:].T)
    
    L = np.linalg.lstsq(A,b,rcond=None)[0]
    # print(f'L:{L}')
    C = np.linalg.cholesky(L)

    motion = motion@C
    structure = np.linalg.inv(C)@structure
    
    return structure

def procrustes_analysis(data1, data2):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    # mtx1 -= np.mean(mtx1, 0)
    # mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    # disparity = np.sum(np.square(mtx1 - mtx2))

    return R, s

def remove_noise(structure):
    new_structure = structure - np.mean(structure, axis=1).reshape((3,1))
    std = np.abs(np.std(new_structure, axis=1))
    length = structure.shape[1]
    to_delete = []
    for i in range(length):
        point = new_structure.T[i]
        if np.abs(point[2] - np.mean(new_structure[2])) > std[2]*2:
            to_delete.append(i)
        if np.abs(point[1] - np.mean(new_structure[1])) > std[1]*2:
            to_delete.append(i)
        if np.abs(point[0] - np.mean(new_structure[0])) > std[0]*2:
            to_delete.append(i)
    structure = np.delete(structure.T, to_delete, axis=0)
    return structure.T

    
def sfm(pvm, N, ambiguity=True, sampling=False, method="icp"):
    pvm = np.where(pvm,pvm,np.nan) 
    structures = np.array([])
    pre_structure = np.array([])
    while pvm.shape[1] >= 3:
        D, pvm = get_dense_submatrix(pvm, N, sampling)
        if not D.size:
            continue 
        if not D.shape[1] > 4:
            continue
        structure = factorization(D, ambiguity, other_decom=False)
        structure = remove_noise(structure)
        if not pre_structure.size:
            pre_structure = structure
            plot_structure((pre_structure))
        else:
            if method == "icp":
                R, t, _ = ICP(structure, pre_structure)
                new_structure = (np.dot(structure.T, R) + t).T
                if not structures.size:
                    structures = pre_structure
                pre_structure = structures
            elif method == "procrustes":
                l1, l2 = pre_structure.shape[1], structure.shape[1]
                if l1 > l2:
                    padding_structure = np.pad(structure, ((0,0),(0,np.abs(l1-l2))),'constant',constant_values = (0,0))
                    R, s = procrustes_analysis(pre_structure, padding_structure)
                    new_structure = np.dot(padding_structure, R.T) * s
                elif l1 < l2:
                    pre_structure = np.pad(pre_structure, ((0,0),(0,np.abs(l1-l2))),'constant',constant_values = (0,0))
                    R, s = procrustes_analysis(pre_structure, structure)
                    new_structure = np.dot(structure, R.T) * s
                else:
                    R, s = procrustes_analysis(pre_structure, structure)
                    new_structure = np.dot(structure, R.T) * s
                if not structures.size:
                    structures = new_structure
                pre_structure = new_structure
            
            structures = np.hstack((structures, new_structure))
            
    return remove_noise(structures)

def plot_matrix(matrix):

    matrix = np.array(matrix)
    plt.matshow(matrix, fignum=100)
    plt.gca().set_aspect('auto')
    plt.show()

    return

def read_matrix(path):

    f = open ('path.txt' , 'r')
    l = []
    l = [ line.split() for line in f]
    matrix = []

    for i in range(len(l)):
        row = [int(float(x)) for x in l[i]]
        matrix.append(row)
    
    return matrix

def improve_density(pvm, N):
    mask = (~np.isnan(pvm)).sum(0) >= N
    return pvm[:, mask]
    
def test_pointViewMatrix(ambiguity):
    pvm = np.loadtxt('../PointViewMatrix.txt').astype('float')
    print(pvm.shape)
    structure = factorization(pvm, ambiguity, other_decom=False)
    print(structure.shape)
    plot_structure(structure)

def eight_point(img1, img2):
    left = cv2.imread(img1, -1)
    right = cv2.imread(img2, -1)

    matches, kp_data1, kp_data2 = match_keypoints(left, right, show_image=False)
    kp1 = np.array([kp_data1[0][match.queryIdx].pt for match in matches])
    kp2 = np.array([kp_data2[0][match.trainIdx].pt for match in matches])

    eight_kp1, eight_kp2 = sample(kp1, kp2)

    nkp1, T1 = norm_kps(kp1)
    nkp2, T2 = norm_kps(kp2)
    eight_nkp1, eight_nkp2 = sample(nkp1, nkp2)

    F = fund_matrix(eight_kp1, eight_kp2)
    F_norm = fund_matrix(eight_nkp1, eight_nkp2)
    F_denorm = T2.T @ F_norm @ T1
    best_kp1, best_kp2, F_rans = ransac(kp1, kp2)

    #Plot regular
    plot(left, right, eight_kp1, eight_kp2, F)

    #Plot normalized
    plot(left, right, eight_kp1, eight_kp2, F_denorm)

    #Plot RANSAC
    plot(left, right, best_kp1, best_kp2, F_rans)

    return

if __name__ == "__main__":
    N = 3 #or 4

    # Part 3
    img1 = 'Data/House/frame00000001.png'
    img2 = 'Data/House/frame00000018.png'
    eight_point(img1, img2)

    # Part 4
    pvm = chaining()
    pvm = np.where(pvm,pvm,np.nan)
    plot_matrix(pvm)

    # Part 5
    structures = sfm(pvm, N, ambiguity=False, sampling=False, method="icp")
    plot_structure(structures)

    # Part 6
    pvm = improve_density(pvm, 10)
    structures = sfm(pvm, N, ambiguity=True, sampling=True, method="icp")
    plot_structure(structures)
     

