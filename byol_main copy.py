import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from glob import glob
import argparse
import torch.nn as nn


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
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = argparse.Namespace()
    args.n_iter = 10000
    args.batch_size = 256
    args.print_interval = 10

    file_paths = glob('PATH_TO_IMAGES/*.jpeg')
    pd.DataFrame(file_paths).to_csv('patches.csv', index=None)
    df = pd.read_csv('patches.csv', header=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # optinal normalization, need to compute these values from some images sampled from the dataset
    # mean = [0.67142541, 0.42631928, 0.67738664] 
    # std = [0.19064334, 0.23075863, 0.15648619]
    image_transforms = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean, std)
    ])

    patch_loader = DataLoader(PatchDataset(df, image_transforms=image_transforms), batch_size=args.batch_size, shuffle=True)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda()
    print(resnet)
    resnet = models.resnet18().cuda()

    learner = BYOL(
        resnet,
        image_size = 256,
        hidden_layer = 'avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    def sample_unlabelled_images():
        return torch.randn(20, 3, 256, 256)

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