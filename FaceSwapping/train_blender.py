import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils
import argparse


# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = 'GAN/data_set/data/'
pretrained_root = 'GAN/Pretrained_model/'
train_list = 'GAN/data_set/train.str'
test_list = 'GAN/data_set/test.str'
batch_size = 4
nthreads = 4
max_epochs = 1
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size

check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
pred_loc = 'Exp_%s/visuals/pred_images' % experiment_name.replace(' ', '_')
source_loc = 'Exp_%s/visuals/source_images' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
os.makedirs(pred_loc, exist_ok=True)
os.makedirs(source_loc, exist_ok=True)

checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)
print(device)
done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
G = res_unet.MultiScaleResUNet(in_nc=7, out_nc=3)
D = discriminators_pix2pix.MultiscaleDiscriminator()
print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.
pretrained_models = os.listdir(pretrained_root)
for m in pretrained_models:
    if 'G' in m:
        model_G, _, iter_count_G = utils.loadModels(G, pretrained_root+m)
    if 'D' in m:
        model_D, _, iter_count_D = utils.loadModels(D, pretrained_root+m)
model_G, _, _ = utils.loadModels(G, 'Exp_Blender/checkpoints_no_D/checkpoint_G_1.pth')
model_G = model_G.to(device)
model_D = model_D.to(device)
print(done)

print('[I] STATUS: Initiate optimizer...', end='')
# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.SGD(G.parameters(), lr=lr_gen, momentum=momentum, weight_decay=weightDecay)
optimizer_D = torch.optim.SGD(D.parameters(), lr=lr_dis, momentum=momentum, weight_decay=weightDecay)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size , gamma=gamma)
print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
loss_gan = gan_loss.GANLoss().to(device)
vgg_loss = vgg_loss.VGGLoss().to(device)
pixelwise = nn.L1Loss()
print(done)

print('[I] STATUS: Initiate Dataloaders...')
# Initialize your datasets here
test_dataset = SwappedDatasetLoader(test_list, data_root)
train_dataset = SwappedDatasetLoader(train_list, data_root)

testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=nthreads)
trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=nthreads)
batches_train = len(trainLoader) / batch_size
batches_test = len(testLoader) / batch_size
print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)

def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1.0 - mask)

def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can use the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    """
    https://github.com/YuvalNirkin/fsgan/blob/master/train_blending.py
    """
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) + np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask, center, cv2.NORMAL_CLONE)
    return output

def alpha_blend(source_img, target_img, mask):
    mask = np.tile(mask, (1,1,3)).astype(float)/255
    source_img = source_img.astype(float)
    target_img = target_img.astype(float)
    foreground = cv2.multiply(mask, source_img)
    background = cv2.multiply(1.0 - mask, target_img)
    outImage = cv2.add(foreground, background)
    return outImage/255

def gaussian_pyramid(img, num_levels):
    img = img.astype(float)
    lower = img.copy()
    gaussian_pyr = [lower.astype(float)]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower.astype(float))
    return gaussian_pyr

def laplacian_pyramid(img, num_levels):
    gaussian_pyr = gaussian_pyramid(img, num_levels)
    lap_top = gaussian_pyr[-1]
    lap_pyr = [lap_top]
    for i in range(len(gaussian_pyr)-1, 0, -1):
        size = (gaussian_pyr[i-1].shape[1], gaussian_pyr[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        lap = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        lap_pyr.append(lap)
    return lap_pyr

def lap_blend(source, target, mask_pyr):
    laplacian_pyr = []
    for l_source, l_target, m in zip(source, target, mask_pyr):
        lap = transfer_mask(l_source, l_target, m)
        laplacian_pyr.append(lap)
    return laplacian_pyr

def reconstruct(lap_pyr, num_levels):
    lap_final = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        size = (lap_pyr[i].shape[1], lap_pyr[i].shape[0])
        lap_final = cv2.add(cv2.pyrUp(lap_final, dstsize=size), lap_pyr[i].astype(float))
        lap_final[lap_final > 255] = 255
        lap_final[lap_final < 0] = 0
    return lap_final/255

def blend_imgs(source_tensor, target_tensor, mask_tensor, blend_func):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        if blend_func == "laplacian":
            source = laplacian_pyramid(source_img, 5)
            target = laplacian_pyramid(target_img, 5)
            mask = np.tile(mask, (1, 1, 3))
            mask_pyr = gaussian_pyramid(mask, 5)
            mask_pyr.reverse()
            laplacian_pyr = lap_blend(source, target, mask_pyr)
            out_bgr = reconstruct(laplacian_pyr, 5)
        mask = np.round(mask * 255).astype('uint8')
        if blend_func == 'poisson':
            out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        if blend_func == 'alpha':
            out_bgr = alpha_blend(source_img, target_img, mask)


        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)

def Train(args, G, D, epoch_count, iter_count):
    torch.cuda.empty_cache()
    G.train(True)
    D.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    Epoch_time = time.time()
    total_loss_pix = 0
    total_loss_id = 0
    total_loss_attr = 0
    total_loss_rec = 0
    total_loss_G_Gan = 0
    total_loss_D_Gan = 0


    for i, data in pbar:
        iter_count += 1
        images, _ = data
        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        G.train()
        D.train()
        with torch.no_grad():
            source_img = images["source"]
            swap_img = images["swap"]
            target_img = images["target"]
            mask_tensor = images["mask"]

            img_transfer = transfer_mask(swap_img, target_img, mask_tensor)
            img_transfer_input = torch.cat((img_transfer, target_img, mask_tensor.float()), dim=1).to(device)
            img_blend = blend_imgs(swap_img, target_img, mask_tensor.float(), args.blend_func).float().to(device)

        # 2) Feed the data to the networks.
        img_blend_pred = G(img_transfer_input)
        img_blend_pred = img_blend_pred.detach()

        # 4) Calculate the losses.
        pred_fake_pool = D(img_blend_pred)

        loss_D_fake = loss_gan(pred_fake_pool, False)
        pred_real = D(target_img.to(device))

        loss_D_real = loss_gan(pred_real, True)

        loss_D_total = (loss_D_fake + loss_D_real) * 0.5

        pred_fake = D(img_blend_pred)
        loss_G_GAN = loss_gan(pred_fake, True)

        # Reconstruction
        loss_pix = pixelwise(img_blend_pred, img_blend)
        loss_id = vgg_loss(img_blend_pred, img_blend)
        loss_attr = vgg_loss(img_blend_pred, img_blend)
        loss_rec = pix_weight * loss_pix + 0.5 * loss_id + 0.5 * loss_attr

        loss_G_total = rec_weight * loss_rec + gan_weight * loss_G_GAN

        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.
        # Update generator weights
        loss_G_total.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()

        # Update discriminator weights
        loss_D_total.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()

        total_loss_pix += loss_pix
        total_loss_id += loss_id
        total_loss_attr += loss_attr
        total_loss_rec += loss_rec
        total_loss_G_Gan += loss_G_GAN
        total_loss_D_Gan += loss_D_total

        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write(f"{args.blend_func}_losses_{epoch_count}_{iter_count}:, pixelwise={loss_pix}, vgg_id={loss_id}, attr={loss_attr}, "
                              f"rec={loss_rec} ,g_gan={loss_G_GAN}, d_gan={loss_D_total}\n")

        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator
    t_source, t_swap, t_target, t_pred, t_blend = Test(args, G)

    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/%s_Epoch_%d_output_%d.png' %
                        (args.blend_func, epoch_count, b), grid)

        source = img_utils.tensor2rgb(t_source[b].detach())
        imageio.imwrite(source_loc + '/%s_source_%d.png' %
                                         (args.blend_func, b), source)
        pred = img_utils.tensor2rgb(t_pred[b].detach())
        imageio.imwrite(pred_loc + "/%s_pred_%d.png" %
                        (args.blend_func, b), pred)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(total_loss_pix.detach().cpu()),\
        np.nanmean(total_loss_id.detach().cpu()), np.nanmean(total_loss_attr.detach().cpu()),\
        np.nanmean(total_loss_rec.detach().cpu()), np.nanmean(total_loss_G_Gan.detach().cpu()),\
        np.nanmean(total_loss_D_Gan.detach().cpu()), iter_count

def Test(args, G):
    with torch.no_grad():
        G.eval()
        pbar = tqdm(enumerate(testLoader), total=batches_test, leave=False)
        # for i, data in pbar:
        #     images, _ = data
        t = enumerate(testLoader)
        i, (images, _) = next(t)
        # Feed the network with images from test set
        source_img = images["source"]
        swap_img = images["swap"]
        target_img = images["target"]
        mask_tensor = images["mask"]

        # 2) Feed the data to the networks.
        img_transfer = transfer_mask(swap_img, target_img, mask_tensor)
        img_transfer_input = torch.cat((img_transfer, target_img, mask_tensor.float()), dim=1).to(device)

        img_blend = blend_imgs(swap_img, target_img, mask_tensor.float(), args.blend_func).float()


        pred = G(img_transfer_input)

            # You want to return 4 components:
            # 1) The source face.
            # 2) The 3D reconsturction.
            # 3) The target face.
            # 4) The prediction from the generator.
            # 5) The GT Blend that the network is targettting.
            # for b in range(t_pred.shape[0]):
            #     source = img_utils.tensor2rgb(source_img[b].detach())
            #     imageio.imwrite(source_loc + '/%s_source_%d_%d.png' %
            #                     (args.blend_func, b, i), source)
            #     pred = img_utils.tensor2rgb(t_pred[b].detach())
            #     imageio.imwrite(pred_loc + "/%s_pred_%d_%d.png" %
            #                     (args.blend_func, b, i), pred)
        return source_img, swap_img, target_img, pred, img_blend


def main(args):
    iter_count = 0
    # Print out the experiment configurations. You can also save these to a file if
    # you want them to be persistent.
    print('[*] Beginning Training:')
    print('\tMax Epoch: ', max_epochs)
    print('\tLogging iter: ', displayIter)
    print('\tSaving frequency (per epoch): ', saveIter)
    print('\tModels Dumped at: ', check_point_loc)
    print('\tVisuals Dumped at: ', visuals_loc)
    print('\tExperiment Name: ', experiment_name)

    for i in range(max_epochs):
        # Call the Train function here
        # Step through the schedulers if using them.
        # You can also print out the losses of the network here to keep track of
        # epoch wise loss.
        loss_pix, loss_id, loss_attr, loss_rec, loss_G_Gan, loss_D_Gan, iter_count = Train(args, G, D, i, iter_count)
        trainLogger.write(f"{args.blend_func}_losses_FINAL: pixelwise={loss_pix}, vgg_id={loss_id}, attr={loss_attr}, "
                          f"rec={loss_rec},g_gan={loss_G_Gan}, d_gan={loss_D_Gan}\n")
        scheduler_G.step()
        scheduler_D.step()

    t_source, t_swap, t_target, t_pred, t_blend = Test(args, G)
    # for b in range(t_pred.shape[0]):
    #     total_grid_load = [t_source[b], t_swap[b], t_target[b],
    #                        t_pred[b], t_blend[b]]
    #     grid = img_utils.make_grid(total_grid_load,
    #                                cols=len(total_grid_load))
    #     grid = img_utils.tensor2rgb(grid.detach())
    #     imageio.imwrite(visuals_loc + '/%s_Epoch_%d_output_%d.png' %
    #                     (args.blend_func, 0, b), grid)
    #     source = img_utils.tensor2rgb(t_source[b].detach())
    #     imageio.imwrite(source_loc + '/%s_source_%d.png' %
    #                                      (args.blend_func, b), source)
    #     pred = img_utils.tensor2rgb(t_pred[b].detach())
    #     imageio.imwrite(pred_loc + "/%s_pred_%d.png" %
    #                     (args.blend_func, b), pred)
    trainLogger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_func', default="poisson", type=str, help='blending function')
    args = parser.parse_args()
    # main(args)
    torch.cuda.empty_cache()
    CUDA_LAUNCH_BLOCKING = 1
    args.blend_func = "laplacian"
    main(args)
    # torch.cuda.empty_cache()
    # args.blend_func = "laplacian"
    # main(args)