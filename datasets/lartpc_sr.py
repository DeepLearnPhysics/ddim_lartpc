from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os

class LARTPC_SR(Dataset):
    def __init__(self, root, transform, split='train', interpolation='nearest'):
        self.root = root
        self.transform = transform
        self.split = split
    
        self.data = {
            # 'LR': np.load(os.path.join(root.format(split=split), 'lr_64.npy')),
            'HR': np.load(os.path.join(root.format(split=split, interpolation=interpolation), 'hr_512.npy')),
            'SR': np.load(os.path.join(root.format(split=split, interpolation=interpolation), 'sr_64_512.npy')),
        }

    def __len__(self):
        return len(self.data['HR'])

    def __getitem__(self, index):
        img_hr = self.data['HR'][index]
        img_sr = self.data['SR'][index]

        img_comb = np.stack([img_hr, img_sr], axis=-1)

        # convert to two-channel PIL image
        img_comb = Image.fromarray(img_comb)

        if self.transform:
            img_comb = self.transform(img_comb)

        img_hr = img_comb[0].unsqueeze(0)  # (1, 512, 512)
        img_sr = img_comb[1].unsqueeze(0)  # (1, 512, 512)

        # For now, we're not using any specific target/label
        target = 0

        return {'HR': img_hr, 'SR': img_sr, 'target': target}