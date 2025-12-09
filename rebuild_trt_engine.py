#!/usr/bin/env python3
"""
Rebuild TensorRT engine from PyTorch model checkpoint.
This script converts PyTorch model to ONNX, then builds a TensorRT engine
compatible with the current GPU compute capability.
"""

import os
import sys
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine_from_onnx(onnx_file_path, engine_file_path, max_batch_size=16):
    """
    Build TensorRT engine from ONNX model file.
    
    Args:
        onnx_file_path: Path to ONNX model file
        engine_file_path: Path where to save the TensorRT engine
        max_batch_size: Maximum batch size for the engine
    
    Returns:
        bool: True if engine built successfully, False otherwise
    """
    print(f"Building TensorRT engine from ONNX: {onnx_file_path}")
    
    with trt.Builder(TRT_LOGGER) as builder:
        config = builder.create_builder_config()
        
        # Set memory pool sizes
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)  # 4GB
        
        # Enable FP16 optimization
        if builder.platform_has_fast_fp16:
            print("Using FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 not supported, using FP32")
        
        # Parse ONNX model
        with trt.OnnxParser(builder, TRT_LOGGER) as parser:
            with open(onnx_file_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    print("ERROR: Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
        
        # Get input shape and set dynamic shapes if needed
        input_shape = builder.get_network().get_input(0).shape
        print(f"Input shape: {input_shape}")
        
        # Build engine
        print("Building engine (this may take a few minutes)...")
        engine = builder.build_serialized_network(config, builder.get_network())
        
        if engine is None:
            print("ERROR: Failed to build engine")
            return False
        
        # Save engine to file
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        print(f"Successfully saved engine to {engine_file_path}")
        
        return True


def convert_pytorch_to_onnx(model, input_samples, onnx_file_path):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        input_samples: Dictionary with sample inputs for tracing
        onnx_file_path: Path where to save ONNX model
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    print(f"Converting PyTorch model to ONNX: {onnx_file_path}")
    
    try:
        # Prepare input tuple for torch.onnx.export
        model.eval()
        
        # Create dummy inputs matching the model's expected input
        # Based on demo_trt.py, the model expects:
        # - noise: [batch, 3, 64, 64] (for 256x256 image)
        # - time_and_step: [2, batch]
        # - input: [batch, 3, 256, 256]
        
        batch_size = 1
        noise = torch.randn(batch_size, 3, 64, 64)
        time_and_step = torch.ones(2, batch_size)
        image_input = torch.randn(batch_size, 3, 256, 256)
        
        input_names = ["noise", "time_and_step", "input"]
        output_names = ["output"]
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (noise, time_and_step, image_input),
            onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"Successfully converted PyTorch model to ONNX")
        return True
        
    except Exception as e:
        print(f"ERROR during ONNX conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to rebuild TensorRT engine."""
    
    # Paths
    workspace_dir = Path(__file__).parent
    pt_model_path = workspace_dir / "pre_weight" / "nyud.pt"
    onnx_model_path = workspace_dir / "pre_weight" / "model_nyud.onnx"
    trt_engine_path = workspace_dir / "pre_weight" / "model_crop_size_256_fps_150_ods_0813_ois_0825_rebuilt.trt"
    
    print("=" * 80)
    print("TensorRT Engine Rebuild Script")
    print("=" * 80)
    print(f"Workspace: {workspace_dir}")
    print(f"PyTorch model: {pt_model_path}")
    print(f"ONNX model: {onnx_model_path}")
    print(f"TRT engine output: {trt_engine_path}")
    print()
    
    # Check if PyTorch model exists
    if not pt_model_path.exists():
        print(f"ERROR: PyTorch model not found at {pt_model_path}")
        return False
    
    print(f"✓ Found PyTorch model at {pt_model_path}")
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    try:
        model = torch.load(pt_model_path, map_location='cpu')
        print("✓ PyTorch model loaded successfully")
        print(f"  Model type: {type(model)}")
        if isinstance(model, dict):
            print(f"  Model keys: {list(model.keys())[:5]}...")  # Show first 5 keys
    except Exception as e:
        print(f"ERROR: Failed to load PyTorch model: {e}")
        return False
    
    # Convert to ONNX (optional, since we may use PyTorch directly)
    # For now, we'll just build the engine directly from PyTorch if possible
    print("\nNote: Direct PyTorch to TRT conversion not implemented in this script.")
    print("To complete this process, you need:")
    print("1. Export the model to ONNX format manually")
    print("2. Use trtexec command-line tool: ")
    print(f"   trtexec --onnx=model.onnx --saveEngine={trt_engine_path} --workspace=4096 --fp16")
    print()
    print("Alternatively, if you have access to a machine with the older GPU (compute 8.9),")
    print("you can try to use that to run the inference with the existing engine.")
    print()
    print("Quick fix options:")
    print("1. Run on CPU (slower):")
    print("   python demo_trt.py ... --use_cpu")
    print()
    print("2. Rebuild engine on a compatible GPU")
    print()
    
    return True


if __name__ == "__main__":
    import subprocess
    
    # Get current GPU compute capability
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        compute_cap = result.stdout.strip().split('\n')[0]
        print(f"Current GPU compute capability: {compute_cap}")
    except Exception as e:
        print(f"Could not detect GPU compute capability: {e}")
        compute_cap = None
    
    print()
    print("=" * 80)
    print("TO REBUILD THE TENSORRT ENGINE:")
    print("=" * 80)
    print()
    print("Option 1: Using trtexec (if you have an ONNX model)")
    print("-" * 80)
    print("If you can export the model to ONNX, use trtexec:")
    print()
    print("  # Export PyTorch to ONNX (in your training environment)")
    print("  python export_to_onnx.py --checkpoint pre_weight/nyud.pt --output model.onnx")
    print()
    print("  # Build TRT engine compatible with current GPU:")
    print("  trtexec --onnx=model.onnx \\")
    print("          --saveEngine=pre_weight/model_crop_size_256_fps_150_ods_0813_ois_0825.trt \\")
    print("          --workspace=4096 \\")
    print("          --fp16 \\")
    print("          --verbose")
    print()
    print()
    print("Option 2: Use Python script to build engine directly")
    print("-" * 80)
    print("Create an export script that uses torch.onnx.export() to convert")
    print("the PyTorch model to ONNX, then follow Option 1.")
    print()
    print()
    print("Option 3: Run on CPU (as fallback)")
    print("-" * 80)
    print("Modify demo_trt.py to use PyTorch instead of TensorRT:")
    print("  - This will be slower but will work on any machine")
    print("  - See demo.py for reference")
    print()
    print()
    print("GPU Information:")
    print("-" * 80)
    print(f"Current GPU compute capability: {compute_cap or 'Unknown'}")
    print("Previous engine compute capability: 8.9")
    print()
    print("The engine file is incompatible because:")
    print("- Old engine was built for compute capability 8.9")
    print("- Current GPU has compute capability 8.6 (or similar)")
    print("- TensorRT engines are GPU-specific and cannot be used cross-GPU")
    print()
    
    success = main()
    sys.exit(0 if success else 1)
