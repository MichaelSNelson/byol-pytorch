import os
import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import concurrent.futures
import pandas as pd
from glob import glob
import argparse
import torch.nn as nn
import numpy as np

class PatchDataset(Dataset):
    def __init__(self, file_list_df, image_transforms=None):
        self.file_list_df = file_list_df
        self.transforms = image_transforms
    def __len__(self):
        return len(self.file_list_df)
    def __getitem__(self, idx):
        file_path = self.file_list_df.iloc[idx][0]
        image = Image.open(file_path)
        if self.transforms:
            image = self.transforms(image)
        return image



def calculate_mean_std(csv_file_path, n_images=500, num_threads=8):
    # read file paths from csv file
    df = pd.read_csv(csv_file_path, header=None)

    # randomly select n_images
    selected_indices = np.random.choice(df.shape[0], n_images, replace=False)
    selected_paths = df.iloc[selected_indices][0].tolist()

    # compute mean and std for each channel using multiple threads
    mean = np.zeros(3)
    std = np.zeros(3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_image, path) for path in selected_paths]
        for future in concurrent.futures.as_completed(futures):
            img_array = future.result()
            mean += np.mean(img_array, axis=(0, 1))
            std += np.std(img_array, axis=(0, 1))

    mean /= n_images
    std /= n_images
    print(mean, std)
    return mean, std

def process_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    img_array = np.array(img) / 255.0 # normalize pixel values between 0 and 1
    return img_array

if __name__ == '__main__':
    print("start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    args = argparse.Namespace()
    args.n_iter = 10000
    args.batch_size = 256
    args.print_interval = 10
    print("glob")
    file_paths = glob('single256overlaps/*/*/*.jpeg')
    print(len(file_paths))
    pd.DataFrame(file_paths).to_csv('patches.csv', index=None)
    df = pd.read_csv('patches.csv', header=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # optinal normalization, need to compute these values from some images sampled from the dataset
    # mean = [0.67142541, 0.42631928, 0.67738664] 
    # std = [0.19064334, 0.23075863, 0.15648619]
    mean, std = calculate_mean_std('patches.csv')
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    patch_loader = DataLoader(PatchDataset(df, image_transforms=image_transforms), batch_size=args.batch_size, shuffle=True)
    #resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)#.cuda()
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)#.cuda()
    #print(resnet)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    resnet = nn.DataParallel(resnet, device_ids=[0, 1,2,3,4])
    resnet.to(device)
    #print(resnet)

    learner = BYOL(
        resnet,
        image_size = 256,
        hidden_layer = 'avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # def sample_unlabelled_images():
    #     return torch.randn(20, 3, 256, 256)

    for i in range(args.n_iter):
        images = next(iter(patch_loader))
        loss = learner(images.cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        if i%args.print_interval==0:
            print(f'Iter {i}, Loss {loss.item()}')

    # save your improved network
    torch.save(resnet.state_dict(), f'./improved-net--feat--iter_{args.n_iter}.pt')