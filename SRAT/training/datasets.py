import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MultiViewDataset(Dataset):
    """
    Dataset class for multi-view super-resolution training with enhanced view handling
    
    This dataset handles different multi-view dataset structures and provides
    consistent view pairs for cross-view consistency training.
    """
    def __init__(self, data_root, scale=4, patch_size=64, augment=True):
        super(MultiViewDataset, self).__init__()
        self.data_root = data_root
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale
        self.augment = augment
        
        # Detect dataset structure and initialize accordingly
        self.scenes = []
        self.view_indices = {}
        
        # Support multiple dataset formats
        if os.path.exists(os.path.join(data_root, 'scenes.txt')):
            # Format 1: scenes.txt listing with separate view folders
            with open(os.path.join(data_root, 'scenes.txt'), 'r') as f:
                self.scenes = [line.strip() for line in f if line.strip()]
            
            for scene in self.scenes:
                scene_dir = os.path.join(data_root, scene)
                if os.path.isdir(scene_dir):
                    views = [f for f in os.listdir(scene_dir) if f.endswith('.png') or f.endswith('.jpg')]
                    self.view_indices[scene] = views
        else:
            # Format 2: Scene subfolders with view files
            for item in os.listdir(data_root):
                scene_dir = os.path.join(data_root, item)
                if os.path.isdir(scene_dir):
                    self.scenes.append(item)
                    views = [f for f in os.listdir(scene_dir) if f.endswith('.png') or f.endswith('.jpg')]
                    self.view_indices[item] = views
                    
        # Validate dataset
        if not self.scenes:
            raise ValueError(f"No valid scenes found in {data_root}")
        
        print(f"Found {len(self.scenes)} scenes with total {sum(len(v) for v in self.view_indices.values())} views")
    
    def __len__(self):
        return sum(len(views) for views in self.view_indices.values())
    
    def __getitem__(self, idx):
        # Map flat index to (scene, view) pair
        scene_idx = 0
        while idx >= len(self.view_indices[self.scenes[scene_idx]]):
            idx -= len(self.view_indices[self.scenes[scene_idx]])
            scene_idx += 1
            
        scene = self.scenes[scene_idx]
        view = self.view_indices[scene][idx]
        
        # Get current view
        hr_img = Image.open(os.path.join(self.data_root, scene, view)).convert('RGB')
        
        # Get another view from same scene for consistency training
        other_views = [v for v in self.view_indices[scene] if v != view]
        if other_views:
            alt_view = random.choice(other_views)
            alt_hr_img = Image.open(os.path.join(self.data_root, scene, alt_view)).convert('RGB')
        else:
            # Fallback if only one view exists
            alt_hr_img = hr_img.copy()
        
        # Process images with cropping and augmentation
        # Get HR patches
        if self.patch_size > 0:
            # Get random crop coordinates
            w, h = hr_img.size
            i = random.randint(0, h - self.patch_size) if h > self.patch_size else 0
            j = random.randint(0, w - self.patch_size) if w > self.patch_size else 0
            
            # Crop HR images
            hr_img = TF.crop(hr_img, i, j, self.patch_size, self.patch_size)
            
            # Try to crop alternative view at same location
            # Note: this is an approximation as views may have different perspectives
            try:
                alt_hr_img = TF.crop(alt_hr_img, i, j, self.patch_size, self.patch_size)
            except:
                # Fallback if crop is out of bounds
                alt_w, alt_h = alt_hr_img.size
                alt_i = random.randint(0, alt_h - self.patch_size) if alt_h > self.patch_size else 0
                alt_j = random.randint(0, alt_w - self.patch_size) if alt_w > self.patch_size else 0
                alt_hr_img = TF.crop(alt_hr_img, alt_i, alt_j, self.patch_size, self.patch_size)
        
        # Apply augmentations if enabled
        if self.augment:
            # Common augmentations for both images to maintain consistency
            # Random horizontal flip
            if random.random() < 0.5:
                hr_img = TF.hflip(hr_img)
                alt_hr_img = TF.hflip(alt_hr_img)
            
            # Random vertical flip
            if random.random() < 0.5:
                hr_img = TF.vflip(hr_img)
                alt_hr_img = TF.vflip(alt_hr_img)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                hr_img = TF.rotate(hr_img, angle)
                alt_hr_img = TF.rotate(alt_hr_img, angle)
        
        # Convert to tensors
        hr_img = TF.to_tensor(hr_img)
        alt_hr_img = TF.to_tensor(alt_hr_img)
        
        # Create LR images with bicubic downsampling
        lr_img = TF.resize(hr_img, (self.lr_patch_size, self.lr_patch_size), 
                           interpolation=Image.BICUBIC)
        alt_lr_img = TF.resize(alt_hr_img, (self.lr_patch_size, self.lr_patch_size), 
                              interpolation=Image.BICUBIC)
        
        return {
            'lr': lr_img, 
            'hr': hr_img, 
            'alt_lr': alt_lr_img, 
            'alt_hr': alt_hr_img,
            'scene': scene,
            'view': view,
            'alt_view': alt_view if other_views else view
        }
