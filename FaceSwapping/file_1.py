import argparse
import seaborn as sns
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchgeometry as tgm
import torch

"""
    QUESSTIONS:  
    - what should we use for fov
    - do we obtain far and near by taking max and min of the z?
    - should we first save the projected G as an image and then find the landmarks or can they directly be obtained 
        from the mesh
"""
sys.path.insert(1, './cv2_2022_assignment3')

from load_data import save_obj, load_img


def plot_structure(structure, color_mean):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    structure = structure.T
    ax.scatter(structure[0], structure[1], structure[2], s=7, marker='.', color=color_mean)
    for ii in range(0,360,90):
        for jj in range(10,100,40):
            ax.view_init(elev=jj, azim=ii)
    plt.show()

def get_vertices(A, B):
    model = "model2017-1_face12_nomouth.h5"
    bfm = h5py.File(model, "r")
    mean_id = torch.FloatTensor(bfm['shape/model/mean'])
    N = int(mean_id.shape[0] / 3)
    mean_id = mean_id.reshape((N, 3))

    color = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((N, 3))

    components_id = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)   
    sample_comp_id = components_id[:, 0:A]
    sample_comp_id = torch.FloatTensor(sample_comp_id).reshape(N, 3, A)
    variance_id = torch.sqrt(torch.FloatTensor(bfm['shape/model/pcaVariance']))

    mean_ex = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    components_ex = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    variance_ex = torch.sqrt(torch.FloatTensor(bfm['expression/model/pcaVariance']))
    mean_ex = torch.FloatTensor(mean_ex).reshape(N, 3)

    sample_comp_ex = components_ex[:, 0:B]
    sample_comp_ex = torch.FloatTensor(sample_comp_ex).reshape((N, 3, B))

    # G = mean_id + (sample_comp_id @ (alpha * variance_id[:30])) + mean_ex + (sample_comp_ex @ (delta * variance_ex[:20]))

    triangle_top = np.asarray(bfm['shape/representer/cells'], dtype=np.float32)[:30]
    id = [mean_id, sample_comp_id, variance_id[:30]]
    ex = [mean_ex, sample_comp_ex, variance_ex[:20]]
    return id, ex, color, triangle_top

def rotations(omega):

    rads = torch.deg2rad(omega)
    Ox, Oy, Oz = torch.unsqueeze(rads[0],-1), torch.unsqueeze(rads[1],-1), torch.unsqueeze(rads[2],-1)
    zeros = torch.zeros(1, requires_grad=True)
    ones = torch.ones(1, requires_grad=True)
    R_x = torch.cat((
                    torch.cat((ones, zeros, zeros)),
                    torch.cat((zeros, torch.cos(Ox), -torch.sin(Ox))),
                    torch.cat((zeros, torch.sin(Ox), torch.cos(Ox)))
                    )).reshape(3,3)
    R_y = torch.cat((
                    torch.cat((torch.cos(Oy), zeros, torch.sin(Oy))),
                    torch.cat((zeros, ones, zeros)),
                    torch.cat((-torch.sin(Oy), zeros, torch.cos(Oy)))
                    )).reshape(3,3)
    R_z = torch.cat((
                    torch.cat((torch.cos(Oz), -torch.sin(Oz), zeros)),
                    torch.cat((torch.sin(Oz), torch.cos(Oz), zeros)),
                    torch.cat((zeros, zeros, ones))
                    )).reshape(3,3)
    R = R_z @ R_y @ R_x
    return R

def ridgid_matrix(omega, tau):

    device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = rotations(omega).to(device)
    tau = tau.reshape(1, 3).to(device)

    T = torch.cat((R.T, tau)).T
    T = torch.cat((T, torch.zeros((1, 4)).to(device)))

    T[3, 3] = 1
    return T

def ridgid_transform(T, vertices):
    homogenous_vertices = torch.ones((len(vertices), 4))
    homogenous_vertices[:, :3] = vertices
    transformed_vertices = T.detach().cpu() @ homogenous_vertices.T

    transformed_vertices = transformed_vertices.T[:, :3] / transformed_vertices.T[:, [-1]]
    return transformed_vertices

def viewport_matrix(w, h, l, b):

    V = torch.FloatTensor([[w/2, 0, 0, (l+w)/2],
                    [0, -h/2, 0, (h+b)/2],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1]])

    return V

def projection_matrix(w, h, n, f):

    aspect_ratio = (w / h)
    fovy = 0.5
    t = np.tan(fovy/2) * n # top
    b = -t  # bottom
    r = t * aspect_ratio # right
    l = -r # left
    assert abs(n - f) > 0
    P = torch.FloatTensor([[(2*n)/(r-l), 0, (r+l)/(r-l), 0],
                    [0, (2*n)/(t-b), (t+b)/(t-b), 0],
                    [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                    [0, 0, -1, 0]])
    return P, r, l, t, b

def pinhole(face, omega, tau):
    # Construct transformation matrix
    tau = tau.unsqueeze(dim=-1)
    omega = tgm.deg2rad(omega) # Convert degrees to radian values
    omega_x = omega[0].unsqueeze(dim=-1)
    omega_y = omega[1].unsqueeze(dim=-1)
    omega_z = omega[2].unsqueeze(dim=-1)
    zeros = torch.zeros(1)
    ones = torch.ones(1)

    Rx = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, torch.cos(omega_x), -torch.sin(omega_x)]),
                torch.stack([zeros, torch.sin(omega_x), torch.cos(omega_x)])
                ]).reshape(3,3)

    Ry = torch.stack([
                torch.stack([torch.cos(omega_y), zeros, torch.sin(omega_y)]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([-torch.sin(omega_y), zeros, torch.cos(omega_y)])
                ]).reshape(3,3)

    Rz = torch.stack([
                torch.stack([torch.cos(omega_z), -torch.sin(omega_z), zeros]),
                torch.stack([torch.sin(omega_z), torch.cos(omega_z), zeros]),
                torch.stack([zeros, zeros, ones])
                ]).reshape(3,3)

    R = torch.mm(Rz, torch.mm(Ry, Rx))
    T = torch.cat([R, tau],  dim=1)
    T = torch.cat([T, torch.stack([zeros, zeros, zeros, ones],  dim=1)])

    # Construct Pprojection and viewpoint matrix
    face_np = face.data.numpy()
    face_np = np.hstack((face_np, np.ones((face_np.shape[0],1))))
    l = np.min(face_np[:,0])
    r = np.max(face_np[:,0])
    b = np.min(face_np[:,1])
    t = np.max(face_np[:,1])
    n = np.min(face_np[:,2])
    f = np.max(face_np[:,2])

    P = np.array([[(2*n)/(r-l), 0, (r+l)/(r-l), 0],
                    [0, (2*n)/(t-b), (t+b)/(t-b), 0],
                    [0, 0, -((f+n)/(f-n)), -((2*f*n)/(f-n))],
                    [0, 0, -1, 0]])

    V = np.array([[(r-l)/2, 0, 0, (r+l)/2],
                    [0, (t-b)/2, 0, (t+b)/2],
                    [0, 0, 1/2, 1/2],
                    [0, 0, 0, 1]])

    # Map 3D face to 2D face
    pi = V.dot(P)
    pi = torch.tensor(pi, dtype=torch.float32)
    ones = torch.ones((face.shape[0],1))
    face = torch.cat([face, ones], dim=1)
    face = torch.mm(T, face.T)
    face_2D = torch.mm(pi, face).T[:,:3]

    return(face_2D)    

def mvp(G, omega, tau):
    device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = ridgid_matrix(omega, tau)
    w = max(G[:, 0]) - min(G[:, 0])
    h = max(G[:, 1]) - min(G[:, 1])
    # !!! not sure if this is the correct way
    n = .1 #min(G[:, 2])
    f = 100 #max(G[:, 2])

    P, r, l, t, b = projection_matrix(w, h, n, f)
    V = viewport_matrix(w, h, l, b)

    homogenous_G = torch.ones((len(G), 4)).to(device)
    homogenous_G[:, :3] = G
    projected_G = V.to(device) @ P.to(device) @ T @ homogenous_G.T
    new_G = projected_G[:3, :] / projected_G[-1, :]

    return new_G.T

def main():
    lm_path= 'supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl'
    lm = np.loadtxt(lm_path).astype(int)
    alpha = torch.FloatTensor(30).uniform_(-1, 1)
    delta = torch.FloatTensor(20).uniform_(-1, 1)
    id, ex, color, faces = get_vertices(30, 20)
    mean_id, sample_comp_id, variance_id = id
    mean_ex, sample_comp_ex, variance_ex = ex
    G = mean_id + (sample_comp_id @ (alpha * variance_id)) + mean_ex + (
                sample_comp_ex @ (delta * variance_ex))
    # plot_structure(vertices, color)
    omega = torch.FloatTensor(args.angles)
    tau = torch.FloatTensor(args.translation)
    T = ridgid_matrix(omega, tau)

    transformed_G = ridgid_transform(T, G)
    print(transformed_G.shape, faces.shape, color.shape)
    save_obj(f'{args.angles}_{args.translation}.obj', transformed_G, faces, color)

    projected_G = mvp(G, omega, tau)
    save_obj('projected.obj', projected_G, faces, color)

    plt.scatter(projected_G.detach().cpu().numpy()[lm, 0], projected_G.detach().cpu().numpy()[lm, 1])
    plt.show()
    plt.savefig('lms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', default=[0,10,0], type=list, help='rotation angles [x, y, z]')
    parser.add_argument('--translation', default=[0,0,-500], type=list, help='translations [x, y, z]')
    args = parser.parse_args()
    plt.style.use('seaborn')
    main()


