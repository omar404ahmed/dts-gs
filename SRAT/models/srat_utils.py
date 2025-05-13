import os
import torch
import torch.nn as nn

def load_srat_model(target_path, fallback_path=None, device='cuda'):
    """
    Load SRAT model with robust fallback handling
    
    Args:
        target_path: Path to SRAT pretrained weights
        fallback_path: Path to SWINIR weights for partial loading
        device: Device to load model onto
    
    Returns:
        Loaded SRAT model
    """
    # Import needs to be here to avoid circular imports
    from models.srat.model_srat import SRAT
    
    # Initialize model
    model = SRAT(
        upscale=4, 
        in_chans=3, 
        img_size=64, 
        window_size=8,
        img_range=1., 
        depths=[6, 6, 6, 6], 
        embed_dim=180, 
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='pixelshuffle', 
        resi_connection='1conv'
    )
    
    # Try loading SRAT weights first
    try:
        if os.path.exists(target_path):
            print(f"Loading SRAT weights from {target_path}")
            state_dict = torch.load(target_path, map_location='cpu')
            param_key = 'params' if 'params' in state_dict else None
            model.load_state_dict(
                state_dict[param_key] if param_key else state_dict, 
                strict=True
            )
            print("Successfully loaded SRAT model")
        elif fallback_path and os.path.exists(fallback_path):
            # Map compatible SWINIR weights to SRAT
            print(f"SRAT weights not found. Loading compatible SWINIR weights from {fallback_path}")
            state_dict = torch.load(fallback_path, map_location='cpu')
            param_key = 'params' if 'params' in state_dict else None
            weights = state_dict[param_key] if param_key else state_dict
            
            # Filter out weights for components not in SRAT
            compatible_dict = {k: v for k, v in weights.items() 
                              if k in model.state_dict() and 
                              model.state_dict()[k].shape == v.shape}
            
            # Log compatibility info
            loaded_keys = set(compatible_dict.keys())
            all_keys = set(model.state_dict().keys())
            missing_keys = all_keys - loaded_keys
            print(f"Loaded {len(loaded_keys)}/{len(all_keys)} layers from SWINIR")
            print(f"Missing keys are primarily in texture-aware components: {len([k for k in missing_keys if 'texture' in k])} texture layers")
            
            # Load compatible weights
            model.load_state_dict(compatible_dict, strict=False)
            print("Initialized with partial weights from SWINIR")
        else:
            print("No pretrained weights found. Using random initialization")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Using random initialization")
        
    model = model.to(device)
    model.eval()
    return model