"""
LFW dataloading
"""
import argparse
import time
import os

import numpy as np
import torch, torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(path_to_folder):
            self.listOfFiles += [os.path.join(dirpath, file) for file in filenames if not file.startswith('.')]

        self.transform = transform
        
    def __len__(self):
        return len(self.listOfFiles)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.listOfFiles[index])

        if self.transform:
            img = self.transform(img)

        return img

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        def show_images(img):
            img = img 
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.savefig('visualization_batch.png')
            plt.show()
    
        dataiter = iter(dataloader)
        images = dataiter.next()
        show_images(torchvision.utils.make_grid(images))
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
