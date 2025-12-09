#!/usr/bin/env python3
"""
Quick fix script to run inference with the correct weights.
This script helps users avoid the "Missing key(s)" error by using verified weights.
"""

import argparse
import sys
from pathlib import Path

def suggest_command(input_dir, out_dir):
    """Suggest the correct demo.py command to run."""
    
    print("="*80)
    print("DiffusionEdge Demo - Quick Fix")
    print("="*80)
    print()
    
    # Check available weights
    weights_dir = Path("./outputs/disloss")
    pt_files = sorted(weights_dir.glob("model-*.pt"), key=lambda x: int(x.stem.split('-')[1]))
    
    if pt_files:
        print(f"Found {len(pt_files)} trained model checkpoints in {weights_dir}")
        latest = pt_files[-1]
        print(f"Latest checkpoint: {latest.relative_to('.')}")
        print()
        
        print("RECOMMENDED COMMAND:")
        print("-" * 80)
        print(f"python3 demo.py \\")
        print(f"    --cfg ./configs/BSDS_sample.yaml \\")
        print(f"    --input_dir {input_dir} \\")
        print(f"    --pre_weight {latest.relative_to('.')} \\")
        print(f"    --out_dir {out_dir} \\")
        print(f"    --bs 8")
        print()
        
        print("This command uses:")
        print(f"  - Config: BSDS_sample.yaml (for 320x320 images)")
        print(f"  - Weights: Latest trained model ({latest.name})")
        print(f"  - Input: {input_dir}")
        print(f"  - Output: {out_dir}")
        print()
        
        # Also show alternative configs
        print("ALTERNATIVE CONFIGS (try if above doesn't work):")
        print("-" * 80)
        configs = ["NYUD_sample.yaml", "BIPED_sample.yaml"]
        for cfg in configs:
            print(f"python3 demo.py --cfg ./configs/{cfg} \\")
            print(f"    --input_dir {input_dir} \\")
            print(f"    --pre_weight {latest.relative_to('.')} \\")
            print(f"    --out_dir {out_dir}")
            print()
    
    else:
        print("ERROR: No trained models found in ./outputs/disloss")
        print()
        print("Available options:")
        print("1. Train a new model using train_cond_ldm.py")
        print("2. Use pre-trained weights if available")
        print("3. Check if weights are in a different location")
        print()
        print("To find all .pt files:")
        print("  find . -name 'model-*.pt' -type f")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate suggested command for demo.py"
    )
    parser.add_argument(
        "--input_dir",
        default="./data/Multicue_split/MDBD_split3/test/imgs",
        help="Input image directory"
    )
    parser.add_argument(
        "--out_dir",
        default="./outputs/demo_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    success = suggest_command(args.input_dir, args.out_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
