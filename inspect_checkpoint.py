#!/usr/bin/env python3
"""
Utility script to inspect PyTorch checkpoint files and identify compatible weights.
This helps diagnose issues when loading model checkpoints for inference.
"""

import torch
import sys
from pathlib import Path
import argparse

def inspect_checkpoint(ckpt_path):
    """Inspect and display information about a checkpoint file."""
    
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        print(f"ERROR: File not found: {ckpt_path}")
        return False
    
    print("=" * 80)
    print(f"Checkpoint File: {ckpt_path}")
    print(f"File size: {ckpt_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)
    print()
    
    try:
        print("Loading checkpoint...")
        data = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return False
    
    print(f"Data type: {type(data)}")
    print()
    
    if isinstance(data, dict):
        print("Keys in checkpoint:")
        for key in data.keys():
            print(f"  - {key}: {type(data[key])}")
        print()
        
        # Inspect each major key
        for key in data.keys():
            value = data[key]
            if isinstance(value, dict):
                print(f"\nContent of '{key}':")
                if len(value) == 0:
                    print("  (empty)")
                else:
                    # Show first few keys
                    sample_keys = list(value.keys())[:5]
                    for k in sample_keys:
                        v = value[k]
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: Tensor {tuple(v.shape)} {v.dtype}")
                        else:
                            print(f"  {k}: {type(v)}")
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more keys")
            elif isinstance(value, torch.Tensor):
                print(f"Content of '{key}': Tensor {tuple(value.shape)} {value.dtype}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"Content of '{key}': {type(value).__name__} = {value}")
            else:
                print(f"Content of '{key}': {type(value)}")
    
    elif isinstance(data, torch.nn.Module):
        print("This is a PyTorch Model (nn.Module)")
        print(f"Model type: {type(data)}")
        state_dict = data.state_dict()
        print(f"Model state dict keys (first 5): {list(state_dict.keys())[:5]}")
        print(f"Total parameters: {sum(p.numel() for p in data.parameters())}")
    
    else:
        print(f"Unknown data type: {type(data)}")
    
    print()
    print("=" * 80)
    
    return True


def find_complete_models(root_dir):
    """Search for complete model checkpoints in a directory."""
    
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"ERROR: Directory not found: {root_dir}")
        return
    
    print(f"Searching for .pt files in: {root_dir}")
    print()
    
    pt_files = list(root_dir.rglob("*.pt"))
    if not pt_files:
        print("No .pt files found")
        return
    
    print(f"Found {len(pt_files)} .pt files:")
    print()
    
    for pt_file in sorted(pt_files):
        relative_path = pt_file.relative_to(root_dir)
        size_mb = pt_file.stat().st_size / 1024 / 1024
        
        try:
            data = torch.load(pt_file, map_location='cpu')
            if isinstance(data, dict):
                keys = list(data.keys())
                # Heuristic: complete models usually have 'model' or 'state_dict'
                is_complete = 'model' in keys or 'state_dict' in keys
                marker = "✓ LIKELY COMPLETE" if is_complete else "? UNKNOWN"
                print(f"{marker} {relative_path}")
                print(f"           Size: {size_mb:.2f} MB, Keys: {keys}")
            else:
                print(f"? {relative_path}")
                print(f"           Size: {size_mb:.2f} MB, Type: {type(data).__name__}")
        except Exception as e:
            print(f"✗ {relative_path} (error: {str(e)[:50]}...)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoint files for model loading issues"
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Path to checkpoint file to inspect"
    )
    parser.add_argument(
        "--find",
        help="Search for all .pt files in a directory",
        type=str,
        metavar="DIR"
    )
    
    args = parser.parse_args()
    
    if args.find:
        find_complete_models(args.find)
    elif args.checkpoint:
        inspect_checkpoint(args.checkpoint)
    else:
        parser.print_help()
        print("\nCommon issues and solutions:")
        print("-" * 80)
        print("\n1. 'Missing key(s) in state_dict' error")
        print("   Solution: Use the correct checkpoint file (should be a complete model,")
        print("   not a VAE or encoder-only checkpoint)")
        print()
        print("2. Finding the right checkpoint:")
        print("   python inspect_checkpoint.py --find ./checkpoints")
        print()
        print("3. Inspect a specific file:")
        print("   python inspect_checkpoint.py ./pre_weight/nyud.pt")


if __name__ == "__main__":
    main()
