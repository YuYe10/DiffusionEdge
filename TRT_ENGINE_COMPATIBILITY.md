# TensorRT Engine GPU Compatibility Issue - Resolution Guide

## Problem

When running `demo_trt.py`, you encounter the following error:

```
[TRT] [E] 6: The engine plan file is generated on an incompatible device, 
expecting compute 8.9 got compute 8.6, please rebuild.
```

This means the TensorRT engine file was built for a GPU with compute capability 8.9 (e.g., RTX 6000 Ada), but your current GPU has compute capability 8.6 (e.g., RTX 4090).

## Why This Happens

- **TensorRT engines are GPU-specific**: They are compiled for a particular GPU architecture (compute capability)
- **Compute capability mismatch**: The `.trt` file is incompatible with your GPU
- **You cannot use a TRT engine across different GPUs**: Each GPU needs its own optimized engine

## Solutions

### Solution 1: Rebuild the TRT Engine (Recommended)

You need to rebuild the TensorRT engine on your machine.

#### Prerequisites

1. **TensorRT installed**: Should already be in your environment
2. **ONNX model or conversion script**: The PyTorch model needs to be exportable to ONNX
3. **trtexec tool**: Comes with TensorRT installation

#### Step-by-Step Rebuild

##### Step 1: Export PyTorch Model to ONNX

First, you need to export the PyTorch model at `pre_weight/nyud.pt` to ONNX format.

Create a file `export_model_to_onnx.py`:

```python
#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

def export_to_onnx(pt_path, onnx_path):
    """Export PyTorch model to ONNX format"""
    
    print(f"Loading model from {pt_path}...")
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    # The checkpoint might be the model directly or in a state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
    elif isinstance(checkpoint, dict):
        model_state = checkpoint
    else:
        model_state = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    print(f"Model keys: {list(model_state.keys())[:5]}...")
    print(f"Exporting to ONNX: {onnx_path}")
    
    # Create dummy inputs matching the inference signature
    # Based on demo_trt.py: noise, time_and_step, input
    batch_size = 1
    noise = torch.randn(batch_size, 3, 64, 64)          # [B, C, H/4, W/4]
    time_and_step = torch.ones(2, batch_size)            # [2, B]
    image_input = torch.randn(batch_size, 3, 256, 256)   # [B, C, H, W]
    
    # Since you likely need the actual model class, this is a placeholder
    # You'll need to implement this based on your model architecture
    print("ERROR: Need to implement model export")
    print("You need to:")
    print("1. Load the actual model class from your codebase")
    print("2. Load the checkpoint weights into the model")
    print("3. Call torch.onnx.export() with the correct inputs")
    
    return False

if __name__ == "__main__":
    pt_path = Path("pre_weight/nyud.pt")
    onnx_path = Path("pre_weight/model_nyud.onnx")
    
    if not pt_path.exists():
        print(f"ERROR: Model not found at {pt_path}")
        sys.exit(1)
    
    success = export_to_onnx(str(pt_path), str(onnx_path))
    sys.exit(0 if success else 1)
```

Or, if you have access to a training script or model loading code, use that to properly load and export the model.

##### Step 2: Build TensorRT Engine with trtexec

Once you have the ONNX model:

```bash
trtexec --onnx=pre_weight/model_nyud.onnx \
        --saveEngine=pre_weight/model_crop_size_256_fps_150_ods_0813_ois_0825.trt \
        --workspace=4096 \
        --fp16 \
        --verbose
```

**Parameters explained:**
- `--onnx=`: Path to ONNX model
- `--saveEngine=`: Where to save the rebuilt engine
- `--workspace=4096`: GPU memory workspace in MB (adjust if needed)
- `--fp16`: Use half precision (FP16) for faster inference
- `--verbose`: Show build progress

This will build an engine compatible with your GPU's compute capability (8.6).

##### Step 3: Run demo_trt.py Again

Once the engine is built:

```bash
python demo_trt.py \
    --input_dir ./data/Multicue_split/MDBD_split1/test/imgs \
    --pre_weight ./pre_weight/model_crop_size_256_fps_150_ods_0813_ois_0825.trt \
    --out_dir ./outputs/demo_trt_outputs
```

### Solution 2: Use the PyTorch Demo (Fallback)

If you cannot rebuild the TensorRT engine, use the standard PyTorch demo instead:

```bash
python demo.py \
    --input_dir ./data/Multicue_split/MDBD_split1/test/imgs \
    --pre_weight ./pre_weight/nyud.pt \
    --out_dir ./outputs/demo_outputs \
    --bs 8
```

This will be slower than TensorRT but will work on any GPU.

### Solution 3: Copy Engine from Compatible GPU

If you have access to a machine with a compute 8.9 GPU:

1. Run the inference on that machine to generate the correct engine
2. Copy the `.trt` file to your machine
3. Use it with `demo_trt.py`

**Note**: The engine generated on the compute 8.9 GPU will not work on compute 8.6 GPU, so this only works in reverse (8.6 engine can potentially run on 8.9 with native compilation, but not guaranteed).

## GPU Compute Capability Reference

| GPU Model | Compute Capability |
|-----------|-------------------|
| RTX 4090, 6000 Ada | 8.9 |
| RTX 4000 SFF Ada | 8.9 |
| RTX 3090, 6000, 4000 | 8.6 |
| RTX A6000 | 8.6 |
| A100 | 8.0 |
| H100 | 9.0 |

Check your GPU:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Troubleshooting

### Issue: trtexec not found
**Solution**: Install TensorRT or add it to PATH
```bash
export PATH=/path/to/tensorrt/bin:$PATH
which trtexec
```

### Issue: ONNX export fails
**Solution**: Ensure the model can be properly loaded and traced:
- Check if your model uses unsupported operations
- Use opset version 14 or 15 for best compatibility
- See PyTorch ONNX export documentation

### Issue: TensorRT build fails
**Solution**: 
- Increase workspace size: `--workspace=8192`
- Check available GPU memory: `nvidia-smi`
- Try using FP32 instead of FP16 (remove `--fp16`)

### Issue: Runtime mismatch errors
**Solution**: Ensure versions match:
```bash
python -c "import tensorrt; print(tensorrt.__version__)"
nvcc --version
nvidia-smi
```

## References

- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [trtexec User Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## Additional Help

For questions or issues:
1. Check the error messages carefully - they usually indicate the root cause
2. Run with `--verbose` flag to see detailed build information
3. Consult the TensorRT documentation for your specific error code
