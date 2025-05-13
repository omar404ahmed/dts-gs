"""
Super-Resolution Aware Transformer (SRAT) - A hybrid model for SRGS

This model combines concepts from SWIN3D and SWINIR to create a specialized
super-resolution model for texture generation in SRGS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Multi-layer Perceptron module with LeakyReLU used in transformer blocks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size (int): Window size
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partitioning.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MemoryEfficientWindowAttention(nn.Module):
    """
    Window-based self attention with memory-efficient implementation.
    Inspired by SWIN3D's attention mechanism but adapted for 2D images.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # Adaptive chunk sizing based on available memory
        self.adaptive_chunking = True
        self.base_chunk_size = 128  # Will be adjusted at runtime

        # Define relative position bias table for enhanced position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Define context-aware position encoding tables (inspired by cRSE)
        self.pos_encoding_q = nn.Parameter(
            torch.zeros(2 * window_size - 1, 2 * window_size - 1, num_heads, head_dim))
        self.pos_encoding_k = nn.Parameter(
            torch.zeros(2 * window_size - 1, 2 * window_size - 1, num_heads, head_dim))
        
        # Initialize position encodings
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        trunc_normal_(self.pos_encoding_q, std=0.02)
        trunc_normal_(self.pos_encoding_k, std=0.02)

        # Create relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Define projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        # Project to query, key, value
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C//num_heads
        
        # Scale query
        q = q * self.scale
        
        # Adaptive chunk sizing based on input size
        if self.adaptive_chunking:
            # Adjust chunk size based on tensor dimensions
            device_mem = torch.cuda.get_device_properties(x.device).total_memory / 1024**3  # GB
            if device_mem > 20:  # High-end GPU
                chunk_size = min(256, N)
            elif device_mem > 10:  # Mid-range GPU
                chunk_size = min(128, N)
            else:  # Lower-end GPU
                chunk_size = min(64, N)
        else:
            chunk_size = self.base_chunk_size
        
        # Memory-efficient attention calculation
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Apply softmax and dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        # This is where we use a more memory-efficient implementation
        # We compute attention in chunks to reduce memory usage
        chunk_size = 128  # Adjust based on your GPU memory
        num_chunks = (N + chunk_size - 1) // chunk_size
        
        output = torch.zeros_like(q)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            
            # Process one chunk at a time
            chunk_attn = attn[:, :, start_idx:end_idx, :]
            output[:, :, start_idx:end_idx, :] = torch.matmul(chunk_attn, v)
        
        # Reshape and project
        x = output.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TextureAwareFeatureExtraction(nn.Module):
    """
    Enhanced texture-aware feature extraction with improved gradient handling
    Inspired by SWIN3D's signal encoding but adapted for image textures.

    
    This module leverages gradient information and attention mechanisms to
    better capture high-frequency details important for super-resolution.
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        
        # Multi-scale texture pattern recognition
        self.texture_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // reduction, kernel_size=k, padding=k//2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // reduction, dim // reduction, kernel_size=k, padding=k//2, 
                          groups=dim // reduction),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // reduction, dim, kernel_size=1)
            ) for k in [1, 3, 5]  # Multi-scale kernels
        ])
        
        # Channel attention with CBAM-style implementation
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention for texture awareness
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Gradient branch with directional awareness
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
        )
        
        # Integration layer
        self.integration = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        # Initialize Sobel and Laplacian filters
        self.__init_filters()
        
    def forward(self, x):
        # Multi-scale texture features
        texture_features = sum(branch(x) for branch in self.texture_net) / len(self.texture_net)
        
        # Channel attention
        channel_weights = self.channel_attention(x)
        
        # Spatial attention based on feature statistics
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mean = torch.mean(x, dim=1, keepdim=True)
        spatial_features = torch.cat([spatial_max, spatial_mean], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)
        
        # Apply attention
        texture_features = texture_features * channel_weights * spatial_weights
        
        # Gradient features for edge preservation
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        laplacian = F.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])
        
        # Gradient magnitude and direction for edge-aware processing
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        gradient_features = self.gradient_branch(grad_mag + laplacian.abs())
        
        # Integrate features with residual connection
        integrated_features = self.integration(torch.cat([texture_features, gradient_features], dim=1))
        
        return x + integrated_features
    
    def __init_filters(self):
        # Standard Sobel filters
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                   dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                   dtype=torch.float32).reshape(1, 1, 3, 3))
        
        # Laplacian filter for second derivatives
        self.register_buffer('laplacian', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                                     dtype=torch.float32).reshape(1, 1, 3, 3))


class AdaptiveFeatureControl(nn.Module):
    """
    Adaptively controls feature density based on texture complexity.
    Inspired by SWIN3D's adaptive density control.
    """
    def __init__(self, dim, threshold=0.1):
        super().__init__()
        
        # Texture complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.threshold = threshold
        
        # Feature enhancement for complex regions
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )
        
        # Feature simplification for smooth regions
        self.simplify = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1)
        )
        
    def forward(self, x):
        # Estimate texture complexity
        complexity = self.complexity_estimator(x)
        
        # Create complexity mask
        B, C, H, W = x.shape
        mask = (complexity > self.threshold).float().expand(B, 1, H, W)
        
        # Apply different processing based on complexity
        enhanced = self.enhance(x)
        simplified = self.simplify(x)
        
        # Combine features
        output = mask * enhanced + (1 - mask) * simplified
        
        return output


class SRATBlock(nn.Module):
    """
    Hybrid Transformer Block for SRAT model combining SWIN3D concepts with SWINIR architecture.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 use_texture_aware=False, use_adaptive_control=False,
                 act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_texture_aware = use_texture_aware
        self.use_adaptive_control = use_adaptive_control
        
        # Adjust window size if necessary
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        # Normalization and attention layers
        self.norm1 = norm_layer(dim)
        self.attn = MemoryEfficientWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # DropPath and MLP layers
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Texture-aware feature extraction (in image space)
        if use_texture_aware:
            self.texture_extract = TextureAwareFeatureExtraction(dim)
        
        # Adaptive feature control for complex textures (in image space)
        if use_adaptive_control:
            self.adaptive_control = AdaptiveFeatureControl(dim)

        # Calculate attention mask for shifted window attention
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        
        # Store shortcut for residual connection
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply cyclic shift if needed
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows and reshape
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Apply self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows and reshape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # Reverse cyclic shift if needed
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Reshape and apply residual connection
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # Apply MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Apply texture-aware processing and adaptive control if enabled
        if self.use_texture_aware or self.use_adaptive_control:
            # Convert to image format for texture processing
            x_img = x.transpose(1, 2).reshape(B, C, H, W)
            
            if self.use_texture_aware:
                x_img = self.texture_extract(x_img)
                
            if self.use_adaptive_control:
                x_img = self.adaptive_control(x_img)
            
            # Convert back to token format
            x = x_img.flatten(2).transpose(1, 2)

        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB) with enhanced texture awareness
    and adaptive feature control.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 use_texture_aware=False, use_adaptive_control=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # Build blocks
        self.blocks = nn.ModuleList([
            SRATBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_texture_aware=use_texture_aware,
                use_adaptive_control=use_adaptive_control
            )
            for i in range(depth)
        ])

        # Residual connection
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

        # Patch embedding and unembedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        residual = x
        for block in self.blocks:
            x = block(x, x_size)
        
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        
        return x + residual


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """
    Image to Patch Unembedding
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class Upsample(nn.Sequential):
    """
    Upsample module for super-resolution.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
            m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SRAT(nn.Module):
    """
    Super-Resolution Aware Transformer (SRAT) - Hybrid model for SRGS

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each SRAT layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction upsampler. 'pixelshuffle'/'pixelshuffledirect'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=180, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, upscale=4, img_range=1., 
                 upsampler='pixelshuffle', resi_connection='1conv'):
        super(SRAT, self).__init__()
        
        # Initialize parameters
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        
        # Define mean for normalization
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build RSTB blocks with texture awareness
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_texture_aware=(i_layer > 0),  # Enable texture-aware processing for deeper layers
                use_adaptive_control=(i_layer > 1)  # Enable adaptive control for even deeper layers
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)

        # Build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        # Upsampling
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.conv_up1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_hr = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]


class UpsampleOneStep(nn.Sequential):
    """
    UpsampleOneStep module (direct upsampling from low to high resolution)
    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


def make_model(args, parent=False):
    """
    Create the SRAT model.
    This function allows SRGS to instantiate the model with the same interface.
    """
    return SRAT(
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
