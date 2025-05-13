#!/usr/bin/env python
"""
Two-stage training pipeline for the SRAT (Super-Resolution Aware Transformer) model.

This script implements a comprehensive training strategy with:
1. First stage: Training on standard SR datasets (DIV2K, Flickr2K)
2. Second stage: Fine-tuning on multi-view datasets for better cross-view consistency

Usage:
    python training_pipeline.py --stage 1  # For first stage training
    python training_pipeline.py --stage 2  # For second stage fine-tuning
"""

import os
import sys
import random
import numpy as np
import time
import logging
from datetime import datetime
from collections import OrderedDict, defaultdict
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

# Import SRAT model
from models.srat.model_srat import SRAT
from training.srat.datasets import MultiViewDataset, StandardSRDataset

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
    ]
)
logger = logging.getLogger('SRAT Training')

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loss functions for both training stages
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class EdgeLoss(nn.Module):
    """Edge Loss for preserving structural information"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel.repeat(img.shape[1], 1, 1, 1), groups=img.shape[1])

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
        diff = current - up
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class CrossViewConsistencyLoss(nn.Module):
    """
    Cross-view consistency loss for multi-view datasets.
    Ensures that features extracted from different views are consistent.
    """
    def __init__(self):
        super(CrossViewConsistencyLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, sr_img1, sr_img2, mask=None):
        # Extract high-level features (simplified implementation)
        # In practice, you might use a pretrained network to extract features
        features1 = F.avg_pool2d(sr_img1, kernel_size=4)
        features2 = F.avg_pool2d(sr_img2, kernel_size=4)
        
        # Calculate consistency loss
        if mask is not None:
            # If a valid mask is provided (for areas that should match)
            mask = F.avg_pool2d(mask, kernel_size=4)
            loss = self.criterion(features1 * mask, features2 * mask) / (mask.mean() + 1e-8)
        else:
            loss = self.criterion(features1, features2)
        
        return loss


class SRATLoss(nn.Module):
    """
    Combined loss function for SRAT model training.
    Different components are used based on the training stage.
    """
    def __init__(self, stage=1):
        super(SRATLoss, self).__init__()
        self.stage = stage
        
        # Basic losses used in both stages
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss()
        
        # Stage 2 specific losses
        if stage == 2:
            self.cross_view = CrossViewConsistencyLoss()
        
        # Loss weights
        self.pixel_weight = 1.0
        self.edge_weight = 0.05
        
        # Stage 2 specific weights
        self.cross_view_weight = 0.2
    
    def forward(self, sr_img, hr_img, alt_sr_img=None, alt_hr_img=None):
        """
        Calculate combined loss
        
        Args:
            sr_img: Super-resolved image
            hr_img: High-resolution ground truth
            alt_sr_img: Super-resolved image from alternative view (stage 2 only)
            alt_hr_img: High-resolution ground truth from alternative view (stage 2 only)
        """
        # Calculate basic losses (used in both stages)
        pixel_loss = self.charbonnier(sr_img, hr_img)
        edge_loss = self.edge(sr_img, hr_img)
        
        # Combine losses with weights
        total_loss = (
            self.pixel_weight * pixel_loss +
            self.edge_weight * edge_loss
        )
        
        # Add stage 2 specific losses
        if self.stage == 2 and alt_sr_img is not None and alt_hr_img is not None:
            # Cross-view consistency loss
            cross_view_loss = self.cross_view(sr_img, alt_sr_img)
            
            # Ensure alternative view also produces good SR results
            alt_pixel_loss = self.charbonnier(alt_sr_img, alt_hr_img)
            
            total_loss += (
                self.cross_view_weight * cross_view_loss +
                self.pixel_weight * 0.5 * alt_pixel_loss
            )
        
        return total_loss


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse)


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM between two images"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate gaussian kernel
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    window = create_window(window_size, img1.size(1)).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def setup_optimizer(model, args):
    """Set up optimizer and scheduler"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75), int(args.epochs * 0.9)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, metrics=None, is_best=False):
    """Save checkpoint"""
    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    
    if metrics:
        state['metrics'] = metrics
    
    # Regular checkpoint
    torch.save(state, f"{save_path}/checkpoint_{epoch}.pth")
    
    # Best model checkpoint
    if is_best:
        torch.save(state, f"{save_path}/best_model.pth")
    
    # SRGS format checkpoint (for direct use in SRGS)
    srgs_compat = {
        'params': model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    }
    torch.save(srgs_compat, f"{save_path}/srat_srgs_compat.pth")
    
    logger.info(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        return 0, {}
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, scaler=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
        for iter_idx, batch in enumerate(dataloader):
            # Get data
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            
            # Stage 2 specific data handling
            if args.stage == 2 and 'alt_lr' in batch:
                alt_lr_img = batch['alt_lr'].to(device)
                alt_hr_img = batch['alt_hr'].to(device)
            else:
                alt_lr_img = None
                alt_hr_img = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if args.amp:
                with autocast():
                    # Forward pass
                    sr_img = model(lr_img)
                    
                    # For stage 2, also process alternative view
                    alt_sr_img = model(alt_lr_img) if alt_lr_img is not None else None
                    
                    # Calculate loss
                    loss = criterion(sr_img, hr_img, alt_sr_img, alt_hr_img)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                # Update weights with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass without mixed precision
                sr_img = model(lr_img)
                
                # For stage 2, also process alternative view
                alt_sr_img = model(alt_lr_img) if alt_lr_img is not None else None
                
                # Calculate loss
                loss = criterion(sr_img, hr_img, alt_sr_img, alt_hr_img)
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    
                optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            pbar.update(1)
            
            # Log every log_interval iterations
            if (iter_idx + 1) % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Iter {iter_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
            # Save sample images
            if iter_idx == 0 and epoch % args.save_images_interval == 0:
                save_sample_images(lr_img, sr_img, hr_img, args.save_dir, epoch, f"train_{iter_idx}")
    
    # Return average loss for the epoch
    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device, args):
    """Validate the model"""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Get data
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            
            # Stage 2 specific data handling
            if args.stage == 2 and 'alt_lr' in batch:
                alt_lr_img = batch['alt_lr'].to(device)
                alt_hr_img = batch['alt_hr'].to(device)
            else:
                alt_lr_img = None
                alt_hr_img = None
            
            # Forward pass
            sr_img = model(lr_img)
            
            # For stage 2, also process alternative view
            alt_sr_img = model(alt_lr_img) if alt_lr_img is not None else None
            
            # Calculate loss
            loss = criterion(sr_img, hr_img, alt_sr_img, alt_hr_img)
            
            # Calculate PSNR and SSIM
            psnr = calculate_psnr(sr_img, hr_img)
            ssim = calculate_ssim(sr_img, hr_img)
            
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_loss += loss.item()
            
            # Save sample validation images
            if batch_idx < 4 and args.epoch % args.save_images_interval == 0:
                save_sample_images(lr_img, sr_img, hr_img, args.save_dir, args.epoch, f"val_{batch_idx}")
    
    # Return average metrics
    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    
    metrics = {"psnr": avg_psnr, "ssim": avg_ssim, "loss": avg_loss}
    return metrics


def save_sample_images(lr_img, sr_img, hr_img, save_dir, epoch, prefix):
    """Save sample images for visualization"""
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Only save the first 4 images in batch
    n_samples = min(4, lr_img.size(0))
    
    for i in range(n_samples):
        # Create a grid of LR, SR, HR images
        lr = lr_img[i].cpu()
        sr = torch.clamp(sr_img[i].cpu(), 0, 1)
        hr = hr_img[i].cpu()
        
        # Resize LR to match SR and HR for visualization
        lr_resized = F.interpolate(lr.unsqueeze(0), size=sr.shape[1:], mode='bicubic', align_corners=False).squeeze(0)
        
        grid = make_grid([lr_resized, sr, hr], nrow=3, padding=5, normalize=True)
        save_image(grid, os.path.join(sample_dir, f"{prefix}_sample_{i}_epoch_{epoch}.png"))
    
    # Also save a combined grid
    combined_lr = F.interpolate(lr_img[:n_samples], scale_factor=4, mode='bicubic', align_corners=False)
    combined_sr = torch.clamp(sr_img[:n_samples], 0, 1)
    combined_hr = hr_img[:n_samples]
    
    combined = torch.cat([combined_lr, combined_sr, combined_hr], dim=0)
    grid = make_grid(combined, nrow=n_samples, padding=5, normalize=True)
    save_image(grid, os.path.join(sample_dir, f"{prefix}_grid_epoch_{epoch}.png"))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="SRAT Training")
    
    # Path args
    parser.add_argument('--data_path', type=str, default='datasets', help='Path to dataset root')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints and results')
    
    # Training args
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Training stage (1 or 2)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'], help='Scheduler')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision training')
    
    # Model args
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--patch_size', type=int, default=64, help='HR patch size for training')
    
    # Data args
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Distributed training args
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--local_rank', type=int, default=0, help='Local process rank')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:23456', help='URL for distributed training')
    
    # Logging and checkpoint args
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint saving interval')
    parser.add_argument('--save_images_interval', type=int, default=5, help='Save sample images interval')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set global rank for logging
    args.global_rank = 0
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"srat_stage{args.stage}_x{args.scale}_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up distributed training if enabled
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.global_rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            logger.info("RANK and WORLD_SIZE not found in environment. Using local_rank and world_size from args.")
            args.global_rank = args.local_rank
            args.world_size = args.world_size
            
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.global_rank,
        )
        logger.info(f"Distributed training initialized with rank {args.global_rank}/{args.world_size}")
        
        # Sync process group
        torch.distributed.barrier()
    
    # Set device
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Save args for reproducibility
    if args.global_rank == 0:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    # Create model
    model = SRAT(
        upscale=args.scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection='1conv'
    )
    model = model.to(device)
    
    # Set up criterion based on training stage
    criterion = SRATLoss(args.stage)
    criterion = criterion.to(device)
    
    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, args)
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler() if args.amp else None
    
    # Load checkpoint if provided
    start_epoch, metrics = 0, {}
    if args.resume:
        start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                   find_unused_parameters=False)
        logger.info(f"Model wrapped with DDP on rank {args.global_rank}")
    
    # Create datasets and data loaders
    if args.stage == 1:
        # Stage 1: Standard SR datasets (DIV2K, Flickr2K)
        train_paths = []
        if os.path.exists(os.path.join(args.data_path, 'DIV2K/train/HR')):
            train_paths.append(os.path.join(args.data_path, 'DIV2K/train/HR'))
        if os.path.exists(os.path.join(args.data_path, 'Flickr2K/train/HR')):
            train_paths.append(os.path.join(args.data_path, 'Flickr2K/train/HR'))
        
        val_paths = []
        if os.path.exists(os.path.join(args.data_path, 'DIV2K/valid/HR')):
            val_paths.append(os.path.join(args.data_path, 'DIV2K/valid/HR'))
        
        logger.info(f"Stage 1 training with datasets: {train_paths}")
        logger.info(f"Stage 1 validation with datasets: {val_paths}")
        
        train_dataset = StandardSRDataset(
            train_paths,
            scale=args.scale,
            patch_size=args.patch_size,
            augment=args.augment
        )
        
        val_dataset = StandardSRDataset(
            val_paths,
            scale=args.scale,
            patch_size=args.patch_size,
            augment=False,
            is_train=False
        )
    else:
        # Stage 2: Multi-view dataset
        logger.info(f"Stage 2 training with multi-view dataset: {os.path.join(args.data_path, 'multi_view')}")
        
        train_dir = os.path.join(args.data_path, 'multi_view/train')
        val_dir = os.path.join(args.data_path, 'multi_view/val')
        
        train_dataset = MultiViewDataset(
            train_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            augment=args.augment
        )
        
        val_dataset = MultiViewDataset(
            val_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            augment=False
        )
    
    # Set up samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # For evaluation only
    if args.evaluate:
        logger.info("Running evaluation only")
        val_metrics = validate(model, val_loader, criterion, device, args)
        logger.info(f"Validation Results - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, Loss: {val_metrics['loss']:.4f}")
        return
    
    # Main training loop
    best_psnr = 0.0 if not metrics else metrics.get('psnr', 0.0)
    args.epoch = start_epoch  # For validation image saving
    
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        # Set current epoch for logging
        args.epoch = epoch
        
        # Set epoch for distributed samplers
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, scaler)
        
        # Validate
        logger.info("Running validation...")
        val_metrics = validate(model, val_loader, criterion, device, args)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        if args.global_rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.6f}, Val PSNR: {val_metrics['psnr']:.2f} dB, Val SSIM: {val_metrics['ssim']:.4f}")
            logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
        
        # Save checkpoint
        if args.global_rank == 0 and (epoch + 1) % args.save_interval == 0:
            is_best = val_metrics['psnr'] > best_psnr
            if is_best:
                best_psnr = val_metrics['psnr']
                logger.info(f"New best PSNR: {best_psnr:.2f} dB")
            
            save_checkpoint(model, optimizer, scheduler, epoch, args.save_dir, val_metrics, is_best)
    
    # Save final model
    if args.global_rank == 0:
        save_checkpoint(model, optimizer, scheduler, args.epochs - 1, args.save_dir, val_metrics, False)
        
        # Save a final model in SRGS compatible format
        srgs_compat = {
            'params': model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        }
        torch.save(srgs_compat, os.path.join(args.save_dir, f"srat_x{args.scale}_final.pth"))
        
        logger.info("Training completed!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
