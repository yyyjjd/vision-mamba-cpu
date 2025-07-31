# Vision Mamba CPU优化版本

本项目已经针对CPU推理进行了优化，参考了VisionMamba2cpu项目的实现方法。

## 主要优化内容

### 1. 环境变量控制
添加了两个关键的环境变量来控制CPU回退行为：

- `CAUSAL_CONV1D_FORCE_FALLBACK=TRUE`: 强制使用纯PyTorch实现的causal_conv1d
- `SELECTIVE_SCAN_FORCE_FALLBACK=TRUE`: 强制使用纯PyTorch实现的选择性扫描

### 2. 代码优化
- 修改了`selective_scan_fn`函数，添加环境变量检查
- 修改了`MambaInnerFn`、`MambaInnerFnNoOutProj`、`BiMambaInnerFn`，添加CUDA可用性检查
- 修改了`mamba_simple.py`中的`use_fast_path`逻辑，在CPU环境下自动禁用
- 修改了`setup.py`文件，支持跳过CUDA构建

### 3. 新增脚本
- `vim/setup_cpu_env.py`: CPU环境配置脚本

## 使用方法

### 1. 环境配置

在运行任何Vim模型之前，设置环境变量：

```bash
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE
```

### 2. 安装依赖

```bash
# 安装PyTorch CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装causal_conv1d（跳过CUDA构建）
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
cd causal-conv1d
pip install -e .
cd ..

# 安装mamba_ssm（跳过CUDA构建）
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE
cd mamba-1p1p1
pip install -e .
cd ..
```

### 3. 运行推理

#### 使用环境配置脚本：
```bash
# 先运行环境配置
python vim/setup_cpu_env.py

# 然后运行推理
python infer_vim_tiny.py
```

#### 使用原始评估脚本：
```bash
python vim/main.py --eval \
--resume path/to/model.pth \
--model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--device cpu \
--batch-size 1
```

### 4. 性能测试

推理脚本已经集成了环境变量设置，直接运行即可：
```bash
python infer_vim_tiny.py
```

## 性能优化原理

### 原始项目的问题：
1. 每次调用都会尝试CUDA内核
2. CUDA不可用时抛出异常，然后回退到参考实现
3. 异常处理增加了额外的开销

### 优化后的改进：
1. 通过环境变量预检查CUDA可用性
2. 直接使用纯PyTorch实现，避免异常处理
3. 在CPU环境下禁用fast_path，使用更稳定的实现

## 适用场景

- 树莓派等ARM设备
- 无GPU的服务器环境
- 边缘计算设备
- 开发和测试环境
- 需要CPU推理的生产环境

## 注意事项

1. 在CPU环境下，推理速度会比GPU版本慢，但比原始项目的CPU实现更快
2. 确保设置了正确的环境变量
3. 如果遇到内存不足的问题，请使用较小的batch size
4. 建议使用Vim-Tiny模型，它是为移动/边缘设备设计的较小模型

## 故障排除

### 1. 依赖安装失败
确保在安装前设置了正确的环境变量：
```bash
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE
```

### 2. 推理速度慢
这是正常的，因为使用的是纯PyTorch实现而非优化的CUDA内核。

### 3. 内存不足
降低batch size到1，并确保有足够的swap空间。

### 4. 模型创建错误
检查是否正确设置了环境变量，并确保所有依赖项都已正确安装。 