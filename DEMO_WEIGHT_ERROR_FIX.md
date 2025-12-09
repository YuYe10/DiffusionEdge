# Demo.py 权重加载错误 - 解决指南

## 问题

运行 `demo.py` 时出现错误：

```
RuntimeError: Error(s) in loading state_dict for LatentDiffusion:
        Missing key(s) in state_dict:
```

## 根本原因

这个错误发生的原因通常是：

1. **权重文件与模型架构不匹配**：传入的 `.pt` 文件不是完整的 LatentDiffusion 模型，而是部分权重（如 VAE 编码器）
2. **指定了错误的权重路径**：使用了 VAE-only 或其他不匹配的权重文件
3. **模型配置不匹配**：权重文件是为不同的模型架构训练的

## 解决方案

### 方案 1：使用正确的完整模型权重（推荐）

完整模型权重应该包含 "model" 或 "state_dict" 键。

```bash
python3 demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-20.pt \
    --out_dir ./outputs/demo_split_3_test \
    --bs 8
```

### 方案 2：诊断权重文件

使用提供的诊断脚本检查权重文件内容：

```bash
# 检查单个文件
python3 inspect_checkpoint.py ./pre_weight/nyud.pt

# 搜索所有权重文件
python3 inspect_checkpoint.py --find ./outputs/disloss
```

### 方案 3：使用数据集特定的配置文件

每个数据集都有特定的配置文件，其中包含正确的权重路径。使用对应的配置：

```bash
# 对于 BSDS 数据集
python3 demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --out_dir ./outputs/demo_split_3

# 对于 NYUD 数据集
python3 demo.py \
    --cfg ./configs/NYUD_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --out_dir ./outputs/demo_split_3

# 对于 BIPED 数据集
python3 demo.py \
    --cfg ./configs/BIPED_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --out_dir ./outputs/demo_split_3
```

## 权重文件说明

### 完整模型权重（用于推理）
这些是完整的 LatentDiffusion 模型，包含所有必需的参数：
- 位置: `./outputs/disloss/model-*.pt`
- 大小: ~1-2 GB
- 包含: 完整模型状态字典

### VAE 权重（第一阶段）
这些仅包含自动编码器权重，**不能用于推理**：
- 位置: `./data/total_edges/results_ae_kl_256x256/model-*.pt`
- 大小: 较小（几百 MB）
- 用途: 用于训练时加载预训练的 VAE 编码器

### 模型特定权重
不同数据集可能有专用的权重：
- BSDS: `./checkpoints/BSDS_swin_unet12_disloss_bs2x8/model-*.pt`
- NYUD: `/data/huang/diffusion_edge/checkpoints/NYUD_swin_unet12_no_resize_disloss/model-*.pt`
- BIPED: `/data/huang/diffusion_edge/checkpoints/BIPED_size320_swin_unet12_no_resize_disloss/model-*.pt`

## 命令行参数

### 必需参数
- `--cfg`: 配置文件路径（例如 `./configs/BSDS_sample.yaml`）
- `--input_dir`: 输入图像目录
- `--out_dir`: 输出目录

### 可选参数
- `--pre_weight`: 预训练权重路径（覆盖配置文件中的路径）
- `--sampling_timesteps`: 采样时间步数（默认：1）
- `--bs`: 批处理大小（默认：8）

## 完整示例命令

```bash
# 使用 BSDS 配置和本地训练的模型
python3 demo.py \
    --cfg ./configs/BSDS_sample.yaml \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-20.pt \
    --out_dir ./outputs/demo_results \
    --bs 8 \
    --sampling_timesteps 5

# 使用默认配置和指定的权重
python3 demo.py \
    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
    --pre_weight ./outputs/disloss/model-15.pt \
    --out_dir ./outputs/demo_split_3_test \
    --bs 8
```

## 常见问题

### Q: 如何知道哪个权重文件是完整的？
A: 使用诊断脚本：
```bash
python3 inspect_checkpoint.py --find ./outputs/disloss
```
标记为 "✓ LIKELY COMPLETE" 的文件是完整的模型。

### Q: 权重文件的大小意味着什么？
A: 
- 完整模型: 通常 1-3 GB
- VAE-only: 通常 < 500 MB
- 大小可以是判断的指标，但要用诊断脚本确认

### Q: 可以用 TRT 引擎权重吗？
A: 不可以。`.trt` 文件是编译的 CUDA 代码，不是 PyTorch 权重。
用 `demo.py` 时必须使用 `.pt` 文件。

### Q: 能跳过权重加载直接推理吗？
A: 不能。模型需要预训练的权重才能产生有意义的结果。
如果没有权重，请：
1. 训练一个新模型
2. 使用预训练的公共模型

## 调试步骤

1. **验证权重文件存在：**
   ```bash
   ls -lh ./pre_weight/nyud.pt
   ```

2. **检查权重文件内容：**
   ```bash
   python3 inspect_checkpoint.py ./pre_weight/nyud.pt
   ```

3. **测试配置文件：**
   ```bash
   python3 -c "
   import yaml
   with open('./configs/BSDS_sample.yaml') as f:
       cfg = yaml.load(f, Loader=yaml.FullLoader)
       print('Model image size:', cfg['model']['image_size'])
       print('Checkpoint path:', cfg['sampler']['ckpt_path'])
   "
   ```

4. **使用详细输出运行：**
   ```bash
   python3 -u demo.py \
       --cfg ./configs/BSDS_sample.yaml \
       --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
       --out_dir ./outputs/debug \
       --bs 1 2>&1 | tee debug.log
   ```

## 更多帮助

- 检查配置文件中的路径是否存在
- 确保权重文件没有损坏（尝试用 `torch.load()` 读取）
- 查看 sample_cond_ldm.py 了解正确的模型初始化方式
- 检查训练日志了解使用的配置
