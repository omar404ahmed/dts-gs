import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.srat.srat_utils import load_srat_model

def integrate_srat_with_srgs(scale=4, device=None):
    """
    Helper function to integrate SRAT with SRGS pipeline
    
    Args:
        scale: Super-resolution scaling factor
        device: Target device for model
        
    Returns:
        Integrated SRAT model ready for SRGS
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model paths
    srat_path = f"./model_zoo/srat/srat_x{scale}.pth"
    swinir_path = f"./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth"
    
    # Load model
    model = load_srat_model(srat_path, swinir_path, device)
    
    # Verify model works with SRGS forward pass pattern
    with torch.no_grad():
        # Create dummy input matching SRGS pattern
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        
        # Add padding as done in SRGS
        window_size = 8
        _, _, h, w = dummy_input.size()
        h_pad = (window_size - h % window_size) % window_size
        w_pad = (window_size - w % window_size) % window_size
        
        # Apply padding
        padded_input = F.pad(dummy_input, (0, w_pad, 0, h_pad), 'reflect')
        
        # Test forward pass
        try:
            output = model(padded_input)
            expected_shape = (1, 3, h * scale, w * scale)
            actual_shape = tuple(output.shape)
            
            if expected_shape == actual_shape:
                print(f"✓ Model produces correct output shape: {actual_shape}")
            else:
                print(f"✗ Shape mismatch: Expected {expected_shape}, got {actual_shape}")
                
            print("Integration test passed!")
            return model
            
        except Exception as e:
            print(f"Integration test failed: {e}")
            print("Please check model compatibility with SRGS")
            raise


def verify_srat_integration():
    """
    Run a quick verification test to ensure SRAT works correctly
    """
    try:
        model = integrate_srat_with_srgs()
        print("SRAT integration verified successfully!")
        return True
    except Exception as e:
        print(f"SRAT integration failed: {e}")
        return False

if __name__ == "__main__":
    verify_srat_integration()