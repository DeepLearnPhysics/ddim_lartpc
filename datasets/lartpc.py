from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os

class LARTPC(Dataset):
    def __init__(self, root, transform, split='train', particle_type='both'):
        self.root = root
        self.transform = transform
        self.split = split
        self.particle_type = particle_type

        if particle_type not in ['showers', 'tracks', 'both']:
            raise ValueError("particle_type must be 'showers', 'tracks', or 'both'")

        self.data = []
        if particle_type in ['showers', 'both']:
            self.data.append(np.load(os.path.join(root, f'larcv_png_64_{split}_showers.npy')))
        if particle_type in ['tracks', 'both']:
            self.data.append(np.load(os.path.join(root, f'larcv_png_64_{split}_tracks.npy')))
        
        self.data = np.concatenate(self.data, axis=0).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        
        if self.transform:
            img = self.transform(img)
        
        # For now, we're not using any specific target/label
        target = 0

        return img, target