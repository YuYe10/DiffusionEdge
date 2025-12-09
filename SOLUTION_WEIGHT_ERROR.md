# 权重加载错误 - 完整解决方案

## 问题总结

运行 `demo.py` 时遇到：
```
RuntimeError: Error(s) in loading state_dict for LatentDiffusion:
Missing key(s) in state_dict:
```

## 根本原因

使用了错误的权重文件。常见错误包括：

1. **用 VAE-only 权重代替完整模型**
   - ❌ `./data/total_edges/results_ae_kl_256x256/model-15.pt` - 只是 VAE 权重
   - ✓ `./outputs/disloss/model-20.pt` - 完整模型权重

2. **权重文件与模型配置不匹配**
   - 不同的图像尺寸、架构等导致状态字典键不匹配

3. **指定了不存在的权重路径**
   - 文件不存在或路径错误

## 快速修复（三种方法）

### 方法 1：一键修复脚本（最简单）

```bash
bash ./run_demo_fixed.sh
```

这将自动找到最新的训练模型并运行推理。

### 方法 2：使用快速建议脚本

```bash
python3 quick_demo_fix.py
```

输出正确的命令，然后复制粘贴运行。

### 方法 3：手动使用正确的权重

找到输出目录中的已训练模型：

```bash
python3 demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-20.pt \
    --out_dir ./outputs/demo_split_3_test \
    --bs 8
```

## 关键改进

我已经改进了 `demo.py` 和相关脚本来提供更好的错误处理：

### 1. 增强的错误消息
现在当权重加载失败时，会显示：
- 详细的错误信息
- 状态字典中的键
- 建议的解决方案

### 2. 文件存在性检查
在尝试加载前验证权重文件存在

### 3. flexible 状态字典加载
支持 `strict=False` 模式来处理部分匹配的权重

### 4. 诊断工具
创建了多个诊断脚本来帮助识别问题：
- `inspect_checkpoint.py` - 检查权重文件内容
- `quick_demo_fix.py` - 快速生成正确命令
- `run_demo_fixed.sh` - 自动化脚本

## 权重文件指南

| 文件位置 | 文件大小 | 用途 | 用于推理? |
|---------|--------|------|----------|
| `./outputs/disloss/model-*.pt` | 1-3 GB | 完整 LatentDiffusion 模型 | ✓ 是 |
| `./data/total_edges/results_ae_kl_256x256/model-*.pt` | <500 MB | VAE 编码器权重 | ✗ 否 |
| `./checkpoints/BSDS_swin_unet12_disloss_bs2x8/model-*.pt` | 1-3 GB | BSDS 数据集特定模型 | ✓ 是 |
| `./pre_weight/nyud.pt` | 1-3 GB | NYUD 数据集预训练模型 | ✓ 是 |

## 排查步骤

如果仍有问题，按以下步骤排查：

### Step 1: 验证权重文件

```bash
python3 inspect_checkpoint.py ./outputs/disloss/model-20.pt
```

输出应该显示：
```
✓ LIKELY COMPLETE
  Keys: ['model', 'optimizer', 'epoch', ...]
```

### Step 2: 验证配置文件

```bash
python3 -c "
import yaml
with open('./configs/BSDS_sample.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    print('Image size:', cfg['model']['image_size'])
    print('Model type:', cfg['model']['model_type'])
"
```

### Step 3: 检查路径

```bash
ls -lh ./outputs/disloss/model-*.pt | head -5
```

### Step 4: 详细日志

```bash
python3 -u demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-20.pt \
    --out_dir ./outputs/debug \
    --bs 1 2>&1 | tee debug.log
```

## 命令参数说明

```
--cfg              配置文件路径（必需）
--input_dir        输入图像目录（必需）
--pre_weight       权重文件路径（必需，必须是完整模型）
--out_dir          输出目录（必需）
--bs               批处理大小（默认：8）
--sampling_timesteps 采样步数（默认：1，可增加获得更好效果）
```

## 成功运行的示例

```bash
# 完整的工作命令
python3 demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-20.pt \
    --out_dir ./outputs/demo_results \
    --bs 8 \
    --sampling_timesteps 5
```

预期输出：
```
Loading checkpoint from: ./outputs/disloss/model-20.pt
✓ Checkpoint loaded successfully with strict=True
made attention of type 'vanilla' with 512 in_channels
...
(inference progress bars)
...
sampling complete
```

## 常见错误及解决

### 错误 1: FileNotFoundError: [Errno 2] No such file or directory

**原因**：权重文件不存在
**解决**：检查文件路径
```bash
ls -l ./outputs/disloss/model-20.pt
```

### 错误 2: Missing key(s) in state_dict

**原因**：使用了错误的权重文件（如 VAE-only）
**解决**：使用 `quick_demo_fix.py` 找到正确的权重

### 错误 3: RuntimeError: mat1 and mat2 shapes cannot be multiplied

**原因**：图像尺寸与模型配置不匹配
**解决**：确保使用对应的配置文件（BSDS_sample.yaml 用于 320x320）

## 更多资源

- 详见 `DEMO_WEIGHT_ERROR_FIX.md` 了解更多细节
- 详见 `TRT_ENGINE_COMPATIBILITY.md` 了解 TensorRT 相关问题

## 获取帮助

1. 运行诊断脚本：
   ```bash
   python3 inspect_checkpoint.py --find ./outputs
   ```

2. 查看日志文件：
   ```bash
   tail -f debug.log
   ```

3. 确认配置和权重匹配：
   ```bash
   # 配置中的图像尺寸
   grep image_size configs/BSDS_sample.yaml
   ```
