import copy

from cv2 import resize

from file_1 import *
from supplemental_code.supplemental_code import *
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dlib
import os
import glob
import cv2
import math
import tqdm as tqdm
from preprocess_img import Preprocess
from load_data import *
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from load_data import save_obj, load_img


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.alpha = torch.FloatTensor(30).uniform_(args.range_alpha[0], args.range_alpha[1])
        self.delta = torch.FloatTensor(20).uniform_(args.range_delta[0], args.range_delta[1])

        self.device = args.device

        self.alpha = Variable(self.alpha.to(self.device), requires_grad=True)
        self.delta = Variable(self.delta.to(self.device), requires_grad=True)
        self.omega = Variable(torch.FloatTensor(args.angles).to(self.device), requires_grad=True)
        self.tau = Variable(torch.FloatTensor(args.translation).to(self.device), requires_grad=True)

        # additional scaling and translation
        self.batchnorm = nn.BatchNorm1d(2)

    def forward(self, id, ex, lms):
        mean_id, sample_comp_id, variance_id = id
        mean_ex, sample_comp_ex, variance_ex = ex

        G = mean_id + (sample_comp_id @ (self.alpha * variance_id)) + mean_ex + (
                sample_comp_ex @ (self.delta * variance_ex))

        face_2D = mvp(G, self.omega, self.tau)

        preds_lms = lm_predicitions(face_2D, lms)
        # preds_lms[:,1] = preds_lms[:,1] * -1

        return preds_lms

def ground_truth_lm(img_path):
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    win = dlib.image_window()
    shape = None
    for f in glob.glob(os.path.join(img_path)):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        #     win.add_overlay(shape)

        # win.add_overlay(dets)
        # dlib.hit_enter_to_continue()
    return shape


def resize_pred(pred, gt):

    ratio = (torch.max(pred[:, 0]) - torch.min(pred[:, 0])) / (torch.max(gt[:, 0]) - torch.min(gt[:, 0]))
    resized_pred = pred/ratio
    if pred.shape[1] == 2:
        trans = torch.mean(gt, axis=0).view((1,2)) - torch.mean(resized_pred, axis=0).view((1,2))
        resized_pred += trans
    else:
        trans = torch.mean(gt, axis=0).view((1,2)) - torch.mean(resized_pred[:,:2], axis=0).view((1,2))
        resized_pred[:,:2] += trans

    return resized_pred


def get_gt(img_path):

    gt = ground_truth_lm(img_path)
    gt = shape_to_np(gt)
    gt = torch.FloatTensor(gt)
    # gt[:,1] = gt[:,1]*-1
    # plt.scatter(gt[:,0], gt[:,1])
    # plt.show()
    gt_mean_x = torch.mean(gt[:, 0])
    gt_mean_y = torch.mean(gt[:, 1])
    # gt[:, 0] -= gt_mean_x/2
    # gt[:, 1] -= gt_mean_y/2

    return gt

def lm_predicitions(face_2D, lms):
    lm_preds = face_2D[lms]
    return lm_preds[:, :2]

def compute_loss(args, gt, pred, alpha, delta):
    l_lan = torch.mean((pred - gt).pow(2))
    l_reg = args.lambda_alpha * torch.sum(alpha.pow(2)) + args.lambda_delta*torch.sum(delta.pow(2))
    l_fit = l_lan + l_reg
    return l_fit

def train(args, id, ex, lms, gts, multi=False):
    models = []

    for image in range(len(gts)):
        model = Model()
        model.train()
        models.append(model)

    mean_id, sample_comp_id, variance_id = id
    mean_ex, sample_comp_ex, variance_ex = ex
    id_device = [mean_id.to(args.device), sample_comp_id.to(args.device), variance_id.to(args.device)]
    ex_device = [mean_ex.to(args.device), sample_comp_ex.to(args.device), variance_ex.to(args.device)]

    optimizer = torch.optim.Adam([{'params': model.omega, 'lr': 1},
                                  {'params': model.tau, 'lr': 1},
                                  {'params': model.alpha, 'lr': 0.1},
                                  {'params': model.delta, 'lr': 0.1}])
    if multi:
        optimizer = torch.optim.Adam([{'params': model.alpha, 'lr': 0.1},
                                     {'params': model.delta, 'lr': 0.1}])
    losses_fit = []

    model.train()
    best_loss = 100000
    best_model = 0
    checkpoint_name = 'best_morph_model'
    losses = [[], []]

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        for i in range(len(models)):
            model = models[i]
            gt = gts[i]
            optimizer.zero_grad()

            pred_lms = model.forward(id_device, ex_device, lms)
            resized_pred = resize_pred(pred_lms, gt)

            if epoch == 0:
                plt.scatter(resized_pred.detach().cpu().numpy()[:, 0], resized_pred.detach().cpu().numpy()[:, 1], label='untrained')

            l_fit = compute_loss(args, gt, resized_pred, model.alpha, model.delta)
            if l_fit < best_loss:
                best_loss = l_fit
                best_model = model
            losses_fit.append(l_fit.item())

            l_fit.backward()
            optimizer.step()
    # print(f'first: {best_model.omega}, {best_model.tau}')
    torch.save(best_model.state_dict(), f'./{checkpoint_name}.pth')
    model.eval()
    final_pred = best_model(id_device, ex_device, lms)
    final_pred = resize_pred(final_pred, gt)
    final_pred = final_pred.detach().cpu().numpy()
    plt.scatter(final_pred[:, 0], final_pred[:, 1], label='pred', marker='x')
    gt = gt.detach().cpu().numpy()
    plt.scatter(gt[:, 0], gt[:, 1], label='ground truth')
    plt.legend()
    plt.show()

    return best_model, model.state_dict(), losses_fit, final_pred

def interpolate(x, y, pixels):

    (x1, y1, q11), (x1, y2, q12), (x2, y1, q21), (x2, y2, q22) = sorted(pixels)
    c = ((x2-x)*(y2-y)*q11 + (x-x1)*(y2-y)*q21 + (x2-x)*(y-y1)*q12 + (x-x1)*(y-y1)*q22) 
    c /= ((x2-x1)*(y2-y1))

    return c

def rescale(preds_lms, gt):
    print(preds_lms)
    range_pred = torch.max(preds_lms[:,0]) - torch.min(preds_lms[:,0])
    min_x, min_y = torch.min(preds_lms[:,0]), torch.min(preds_lms[:,1])
    range_gt =  torch.max(gt[:,0]) - torch.min(gt[:,0])
    ratio = range_pred/range_gt
    print(min_x, min_y)
    print(f'ratio11111: {ratio}')
    print(torch.mean(gt, axis=0).reshape(1,2))
    # preds_lms[:,:2] -= torch.mean(gt, axis=0).view((1,2))
    preds_lms[:,0] = (preds_lms[:,0]-min_x)/ratio + min_x #+ torch.mean(gt, axis=0).reshape((1,2))
    preds_lms[:,1] = (preds_lms[:,1]-min_y)/ratio + min_y #+ torch.mean(gt, axis=0).reshape((1,2))
    print(preds_lms)
    return preds_lms


def get_texture_color(img, lm, gt, model, filename, flag, resize=True, all_model=None):
    # img[:,1] = img[:,1]*-1
    id, ex, _, triangle_top = get_vertices(30, 20)

    mean_id, sample_comp_id, variance_id = id
    mean_ex, sample_comp_ex, variance_ex = ex

    if all_model == None:
        face_3D = mean_id + (sample_comp_id @ (model.alpha * variance_id)) + mean_ex + (
                    sample_comp_ex @ (model.delta * variance_ex))
    else:
        face_3D = mean_id + (sample_comp_id @ (all_model.alpha * variance_id)) + mean_ex + (
                    sample_comp_ex @ (all_model.delta * variance_ex))

    if flag:
        # for clinton
        # face_2D = mvp(face_3D, torch.FloatTensor([ 2.5654, -8.4284,  4.4768]), torch.FloatTensor([ -12.5571,  -38.4465, -524.8998]))
        face_2D = mvp(face_3D, torch.FloatTensor([ 6.0190,  6.4990, -0.6959]), torch.FloatTensor([  -4.7474,  -29.4020, -531.8251]))
        
    else:
        face_2D = mvp(face_3D, model.omega, model.tau)
    # face_2D[:,1] *= -1
    print(f'face_2d: {face_2D}')      
    print(f'lm:{lm}')
    if resize:
        pred = resize_pred(face_2D[:,:2].detach(), gt)

        # min_x = torch.min(torch.min(gt[:,0], torch.min(pred[:,0])))
        # min_y = torch.min(torch.min(gt[:,1], torch.min(pred[:,1])))

        # if min_x < 0:
        #     pred[:,0] -= min_x
        #     gt[:,0] -= min_x
        #     face_3D[:,0] -= min_x
        #     face_2D[:,0] -= min_x

        # if min_y < 0:
        #     pred[:,1] -= min_y
        #     gt[:,1] -= min_y
        #     face_3D[:,1] -= min_y
        #     face_2D[:,1] -min_y
    else:
        pred = face_2D[:,:2].detach()
    print(f'pred: {pred}')
    

    plt.scatter(pred[lm][:,0], pred[lm][:,1], marker='x', c='tab:blue',label='prediction')
    plt.scatter(gt[:,0], gt[:,1], marker='o', c='tab:red',label='ground truth')
    plt.legend()
    plt.savefig('result/texturize_{}.png'.format(filename))
    plt.show()
    
    new_color = []

    for p in pred:
        p_color = []
        for i in range(3):
            p0, p1 = [math.floor(x) for x in p]
            try:
                img_color = [(p1, p0, img[p1][p0][i]), (p1, p0+1, img[p1][p0+1][i]),
                        (p1+1, p0, img[p1+1][p0][i]), (p1+1, p0+1, img[p1+1][p0+1][i])]
                p_color.insert(0, interpolate(p1, p0, img_color))
            except:
                print(f'IndexError: index {p0} or {p1} is out of bounds for axis 0 with size')
                p_color.insert(0, 0)
            # print(p_color)
        new_color.append(np.array(p_color) / 255)
    print(np.array(new_color).shape)

    # Scale back
    face_2D = rescale(face_2D, gt)
    pred = torch.hstack((pred, torch.ones(pred.shape[0], 1)))
    # print(pred.shape, np.array(new_color).shape, triangle_top.T.shape)
    mean_point_x = int(torch.mean(face_2D[:,0]))
    mean_point_y = int(torch.mean(face_2D[:,1]))
    shift = np.array([112-mean_point_x, 112-mean_point_y, 0]).reshape((-1,1))
    image = render((face_2D.detach().numpy().T+shift).T, np.array(new_color), triangle_top.T, H=224, W=224)
    plt.imshow(image)
    plt.savefig('result/2D_texture_{}.png'.format(filename))
    plt.show()
    save_obj('object/2D_texture_{}.obj'.format(filename), face_2D, np.array(new_color), triangle_top.T)
    save_obj('object/3D_texture_{}.obj'.format(filename), face_3D, np.array(new_color), triangle_top.T)

    return  face_3D, np.array(new_color), triangle_top

def texturize(img, lm, gt, model, filename, flag=True, all_model=None):
    color = get_texture_color(img, lm, gt, model, filename, flag, all_model)

    # G[:,1] = G[:,1]*-1
    # plt.scatter(G[:,0].detach().numpy(), G[:,1].detach().numpy())
    # plt.scatter(gt[:,0], gt[:,1])
    # plt.show()
                # (file_path, shape, color, triangles)
    # print(G.shape, color.shape, faces.shape)
    # save_obj(f'texturized_new.obj', G.detach().numpy(), color, faces.T)

    return 0

def multiple_frames(list_of_img_paths):
    gts = []
    # get all ground_truths
    for path in list_of_img_paths:
        # image = cv2.imread(path)
        gts.append(torch.FloatTensor(get_gt(path)))
    
    return gts

def crop_image(img, landmarks):
    # calculate 5 facial landmarks using 68 landmarks
    lm3D = load_lm3d()
    
    lm5 = np.zeros((5, 2), dtype=np.float32)
    lm5[0] = (landmarks[36] + landmarks[39]) / 2
    lm5[1] = (landmarks[42] + landmarks[45]) / 2
    lm5[2] = landmarks[30]
    lm5[3] = landmarks[48]
    lm5[4] = landmarks[54]
    input_img, lm_new, transform_params = Preprocess(Image.fromarray(img[...,::-1]), lm5, lm3D)
    return input_img, lm_new

def shift_y(input, min_x, max_x, min_y, max_y):
    return input*224*(max_x-min_x)/((max_y-min_y)*(min_x+max_x+2))

def face_swap(img1, img2, lm, gts, model, filename, flag=False, resize=False):
    G_1, color_1, triangle_1 = get_texture_color(img1, lm, gts[0], model, filename[0], flag)
    G_2, color_2, triangle_2 = get_texture_color(img2, lm, gts[1], model, filename[1], flag)
    face_2D = mvp(G_2, model.omega, model.tau).detach()
    face_2D = rescale(face_2D, gts[1])
    image2 = render(face_2D.numpy(), np.array(color_1), triangle_2.T, H=500, W=500)
    with open('image2.npy', 'wb') as f:
        np.save(f, np.array(image2))
    # cv2.imshow("Image", image2[..., ::-1])
    min_x, max_x = int(torch.min(face_2D[:,0])), int(torch.max(face_2D[:,0]))
    min_y, max_y = int(torch.min(face_2D[:,1])), int(torch.max(face_2D[:,1]))

    print(min_x, max_x, min_y, max_y)
    face_2D[:,0] = face_2D[:,0]*224/(min_x+max_x+4)
    if np.abs(max_y - min_y) - np.abs(max_x - min_x) > 100:
        face_2D[:,1] = shift_y(face_2D[:,1], min_x, max_x, min_y, max_y) - shift_y((min_y+max_y)/2, min_x, max_x, min_y, max_y) + 224/2
    else:
        face_2D[:,1] = face_2D[:,1]*224/(min_y+max_y)
    image = render(face_2D.numpy(), np.array(color_1), triangle_2.T, H=224, W=224)
    print(image.shape)
    # plt.imshow(image)
    # plt.show()
    with open('image.npy', 'wb') as f:
        np.save(f, np.array(image))
    cv2.imshow("Image", image[..., ::-1])
    # cv2.waitKey(0)
    # cv2.imencode('.jpg', image)[1].tofile('lm_image.jpg')
    mask = np.tile(np.any(image > 0, axis=2, keepdims=True), (1, 1, 3))
    bg = np.copy(img2[..., ::-1])
    bg[mask] = 0
    # Sets all foreground pixel to 0
    plt.imshow(bg)
    plt.show()
    print(mask.shape)
    print(bg.shape)
    # bg[mask] = image[mask]
    output = np.where(mask, image, bg/255)
    plt.imshow(output)
    plt.show()

def face_swap_video(img1, img2, lm, gts, model, filename, flag=False):
    G_1, color_1, triangle_1 = get_texture_color(img1, lm, gts[0], model, filename[0], flag)
    G_2, color_2, triangle_2 = get_texture_color(img2, lm, gts[1], model, filename[1], flag)
    face_2D = mvp(G_2, model.omega, model.tau).detach()
    face_2D = rescale(face_2D, gts[1])
    min_x, max_x = int(torch.min(face_2D[:,0])), int(torch.max(face_2D[:,0]))
    min_y, max_y = int(torch.min(face_2D[:,1])), int(torch.max(face_2D[:,1]))

    face_2D[:,0] = face_2D[:,0]*224/(min_x+max_x+14)
    face_2D[:,1] = face_2D[:,1]*224/(min_y+max_y)
    image = render(face_2D.numpy(), np.array(color_1), triangle_2.T, H=224, W=224)
    mask = np.tile(np.any(image > 0, axis=2, keepdims=True), (1, 1, 3))
    bg = np.copy(img2[..., ::-1])
    bg[mask] = 0
    output = np.where(mask, image, bg/255)
    cv2.imwrite(f'video/new_frames/{filename}.jpg', output)
    plt.imshow(output)
    plt.show()

def extract_frames_from_video(path, millisec=1000):
    vidcap = cv2.VideoCapture(path + '/trump.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*millisec))
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        if not success:
            break
        cv2.imwrite("video/frames/frame_%d.jpg" % count, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
    print('FINISH')

def main():
    cropped = True
    id, ex, _, _ = get_vertices(30, 20)
    lm_path = 'supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl'
    lms = np.loadtxt(lm_path).astype(int)
    if args.frame == 'single':
        img_path_1 = 'handsome.jpg'
        img_path_2 = 'pretty.jpg'
        gts = multiple_frames([img_path_1, img_path_2])
        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)
        if cropped:
            fg_cropped, fg_cropped_lm = crop_image(img1, gts[0])
            bg_cropped, bg_cropped_lm = crop_image(img2, gts[1])
            fg_cropped = np.squeeze(fg_cropped)
            bg_cropped = np.squeeze(bg_cropped)
            print(np.squeeze(bg_cropped).shape)
            cv2.imwrite('fg_cropped.jpg', fg_cropped)
            cv2.imwrite('bg_cropped.jpg', bg_cropped)
            gts = multiple_frames(['fg_cropped.jpg', 'bg_cropped.jpg'])
            gt = torch.FloatTensor(get_gt('bg_cropped.jpg')).to(args.device)
            if flag:
                pre_load_model = Model()
                pre_load_model.load_state_dict(torch.load('best_morph_model.pth', map_location=torch.device('cpu'), ))
            else:
                model, state, losses_fit, final_pred = train(args, id, ex, lms, [gt])
                print(f'second: {model.omega}, {model.tau}')
                pre_load_model = model
            
            # # #Perform texturing
            # texturize(img1, lms, gt, pre_load_model, 'clinton')
            # Face swap
            filename = ['handsome', 'pretty']
            face_swap(fg_cropped, bg_cropped, lms, gts, pre_load_model, filename, flag)
        else:
            gt = torch.FloatTensor(get_gt(img_path_1)).to(args.device)
            print(f'gt1:{gt.shape}')
            model, state, losses_fit, final_pred = train(args, id, ex, lms, [gt])
            print(f'second: {model.omega}, {model.tau}')
            pre_load_model = model
            # # #Perform texturing
            # pre_load_model = Model()
            # pre_load_model.load_state_dict(torch.load('best_morph_model.pth', map_location=torch.device('cpu'), ))
            # print(f'third: {pre_load_model.omega}, {pre_load_model.tau}')
            # texturize(img1, lms, gt, pre_load_model, 'clinton')

            # Face swap
            face_swap(img1, img2, lms, gts, pre_load_model)
        
    elif args.frame == 'multiple':
        path = 'multiple'
        files = os.listdir(path)
        gts = []
        for i in range(len(files)):
            image_path = path + f'/frame_{i}.jpg'
            frame = cv2.imread(image_path)
            gt = torch.FloatTensor(get_gt(image_path)).to(args.device)
            frame, frame_lm = crop_image(frame, gt)
            cv2.imwrite(path+f'/frame_cropped_{i}.jpg', frame)
            gt = torch.FloatTensor(get_gt(path+f'/frame_cropped_{i}.jpg')).to(args.device)
            gts.extend(gt)
        all_model, state, losses_fit, final_pred = train(args, id, ex, lms, gts, True)
        for i in range(len(files)):
            print(f'Frame {i}')
            frame = cv2.imread(path+f'/frame_cropped_{i}.jpg')
            gt = torch.FloatTensor(get_gt(path+f'/frame_cropped_{i}.jpg')).to(args.device)
            model, state, losses_fit, final_pred = train(args, id, ex, lms, gts)
            texturize(frame, lms, gt, model, f'frame_{i}', False, all_model)
    elif args.frame == 'video':
        path = 'video'
        files = os.listdir(path + '/frames')
        extract_frames_from_video(path)
        for i in range(len(files)):
            if i == 0:
                fg_img_path = 'handsome.jpg'
                fg_img = cv2.imread(fg_img_path)
                fg_gt = torch.FloatTensor(get_gt(fg_img_path)).to(args.device)
                fg_cropped, fg_cropped_lm = crop_image(fg_img, fg_gt)
                fg_cropped = np.squeeze(fg_cropped)
                cv2.imwrite(path +'/cropped_frames/fg_cropped.jpg', fg_cropped)

            bg_img_path = path + f'/frames/frame_{i}.jpg'
            print(f'Frame {i}')
            bg_gt = torch.FloatTensor(get_gt(bg_img_path)).to(args.device)
            bg_img = cv2.imread(bg_img_path)
            bg_cropped, bg_cropped_lm = crop_image(bg_img, bg_gt)
            bg_cropped = np.squeeze(bg_cropped)
            print(np.squeeze(bg_cropped).shape)
            cv2.imwrite(path + f'/cropped_frames/bg_cropped_{i}.jpg', bg_cropped)
            gts = multiple_frames([path +'/cropped_frames/fg_cropped.jpg', path + f'/cropped_frames/bg_cropped_{i}.jpg'])
            gt = torch.FloatTensor(get_gt(path + f'/cropped_frames/bg_cropped_{i}.jpg')).to(args.device)
            model, state, losses_fit, final_pred = train(args, id, ex, lms, [gt])
            face_swap_video(fg_cropped, bg_cropped, lms, gts, model, f'frame_{i}', flag)
            # texturize(frame, lms, gt, model, f'frame_{i}', False)


    # # Face swap
    # face_swap(img1, img2, lms, gts, pre_load_model)



    # plt.plot(losses_fit, label='fit')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', default=[0, 10, 0], type=list, help='rotation angles [x, y, z]')
    parser.add_argument('--translation', default=[0, 0, -500], type=list, help='translations [x, y, z]')
    parser.add_argument('--range_alpha', default=[-1, 1], type=list, help='range of alpha')
    parser.add_argument('--range_delta', default=[-1, 1], type=list, help='range of delta')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=int, default=0.1, help='Learning rate')
    parser.add_argument('--lambda_alpha', default=0, type=int, help='Regularization on alpha weights')
    parser.add_argument('--lambda_delta', default=0, type=int, help='Regularization on delta weights')
    parser.add_argument('--frame', default='single', type=str, help='choice:single, multiple, video')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    plt.style.use('seaborn')
    flag=False
    main()