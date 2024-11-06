import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch, is_jpg_file
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        
        isSame = True
        for i in range(len(clean_files)):
            if not (clean_files[i].split('.')[0] == noisy_files[i].split('.')[0] == mask_files[i].split('.')[0]):
                print(f"index = {i}, name = {clean_files[i].split('.')[0], noisy_files[i].split('.')[0], mask_files[i].split('.')[0]}")
                isSame = False
                break
        assert isSame
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if (is_png_file(x) or is_jpg_file(x))]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if (is_png_file(x) or is_jpg_file(x))]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if (is_png_file(x) or is_jpg_file(x))]
        assert len(self.clean_filenames) == len(self.noisy_filenames) and len(self.noisy_filenames) == len(self.mask_filenames)
        

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], aug=True)
        mask = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        mask = mask[r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        
        ############## data aug ############
        p = 0.3
        hue_interval = [-p, p]
        saturation_interval = [1-p, 1+p]
        fn_ids = torch.randperm(2)
        h_value = torch.empty(1).uniform_(hue_interval[0], hue_interval[1])
        s_value = torch.empty(1).uniform_(saturation_interval[0], saturation_interval[1])
        for fn_id in fn_ids:
            if fn_id == 0:
                clean = TF.adjust_saturation(clean, s_value)
                noisy = TF.adjust_saturation(noisy, s_value)
            if fn_id == 1:
                clean = TF.adjust_hue(clean, h_value)
                noisy = TF.adjust_hue(noisy, h_value)
        
        
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        
        
        isSame = True
        for i in range(len(clean_files)):
            if not (clean_files[i].split('.')[0] == noisy_files[i].split('.')[0] == mask_files[i].split('.')[0]):
                print(f"index = {i}, name = {clean_files[i].split('.')[0], noisy_files[i].split('.')[0], mask_files[i].split('.')[0]}")
                isSame = False
                break
        assert isSame


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if (is_png_file(x) or is_jpg_file(x))]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if (is_png_file(x) or is_jpg_file(x))]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if (is_png_file(x) or is_jpg_file(x))]
        
        assert len(self.clean_filenames) == len(self.noisy_filenames) and len(self.noisy_filenames) == len(self.mask_filenames)
        


        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], aug=False)
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename, mask_filename

if __name__ == '__main__':
    print(transforms_aug)