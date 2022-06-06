from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import os
from PIL import Image
import pandas as pd


# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, prefix, resize=256):
        self.prefix = prefix
        self.resize = resize

        # Define your initializations and the transforms here. You can also
        # define your tensor transforms to normalize and resize your tensors.
        # As a rule of thumb, put anything that remains constant here.

        with open(data_file, 'r') as f:
            img_names = f.read().splitlines()
        self.data_paths = [os.path.join(self.prefix + p) for p in img_names]
        self.img_idx = [img_i[:5] for img_i in img_names]
        self.transform = transforms.Compose([
                                       transforms.Resize(self.resize),
                                       transforms.CenterCrop(self.resize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        self.transform_mask = transforms.Compose([
                                       transforms.Resize(self.resize),
                                       transforms.CenterCrop(self.resize),
                                       transforms.ToTensor(),
                                   ])


    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return len(self.data_paths)

    def __getitem__(self, index):
        # Write your data loading logic here. It is much more efficient to
        # return your data modalities as a dictionary rather than list. So you
        # can return something like the follows:
        #     image_dict = {'source': source,
        #                   'target': target,
        #                   'swap': swap,
        #                   'mask': mask}

        #     return image_dict, self.data_paths[index]
        if torch.is_tensor(index):
            index = index.tolist()
        img_paths = self.img_idx[index]

        image_files = [file for file in os.listdir(self.prefix) if file.startswith(img_paths)]

        image_dict = {}
        for file in image_files:
            if "fg" in file:
                image_dict["source"] = Image.open(self.prefix + file)
            if "bg" in file:
                image_dict["target"] = Image.open(self.prefix + file)
            if "sw" in file:
                image_dict["swap"] = Image.open(self.prefix + file)
            if "mask" in file:
                image_dict["mask"] = Image.open(self.prefix + file)

        if self.transform:
            image_dict["source"] = self.transform(image_dict["source"])
            image_dict["target"] = self.transform(image_dict["target"])
            image_dict["swap"] = self.transform(image_dict["swap"])
            image_dict["mask"] = self.transform_mask(image_dict["mask"])

        return image_dict, self.data_paths[index]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = './GAN/data_set/data/'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    test_list = './GAN/data_set/test.str'
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images) = next(enu)
        b = time.time()
        # Uncomment to use a prettily printed version of a dict returned by the
        # dataloader.
        # printTensorList(images[0], True)
        print('[*] Time taken: ', b - a)
