import torch
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis
import time
import sys
import os

# 设置CPU环境变量，强制使用fallback实现
os.environ["CAUSAL_CONV1D_FORCE_FALLBACK"] = "TRUE"
os.environ["SELECTIVE_SCAN_FORCE_FALLBACK"] = "TRUE"

# 1. 导入模型注册函数
sys.path.append(os.path.dirname(__file__))  # 保证能import本地模块

# 添加mamba-1p1p1路径，确保能找到mamba_ssm包
mamba_path = os.path.join(os.path.dirname(__file__), 'mamba-1p1p1')
if os.path.exists(mamba_path):
    sys.path.insert(0, mamba_path)
    print(f"[INFO] Added mamba path: {mamba_path}")

from models_mamba import (
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
)

# 配置
MODEL_TYPE = 'vim_small'  # 可选: 'vim_tiny', 'vim_small'

# 根据模型类型设置配置
if MODEL_TYPE == 'vim_tiny':
    CKPT_PATH = 'vim_t_midclstok_76p1acc.pth'  # vim_tiny权重路径
    MODEL_FUNC = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    EMBED_DIM = 192
    DEPTH = 24
    D_STATE = 16
elif MODEL_TYPE == 'vim_small':
    CKPT_PATH = 'vim_s_midclstok_80p5acc.pth'  # vim_small权重路径 - 请根据实际文件名调整
    MODEL_FUNC = vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    EMBED_DIM = 384
    DEPTH = 24
    D_STATE = 16
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

DEVICE = 'cpu'  # 树莓派建议用cpu
BATCH_SIZE = 8
NUM_IMAGES = 100  # 可改为1000
input_size = 224

print(f"[INFO] Using model: {MODEL_TYPE}")
print(f"[INFO] Embed dim: {EMBED_DIM}, Depth: {DEPTH}, D_state: {D_STATE}")

# 2. 加载模型
model = MODEL_FUNC(num_classes=1000)
model.eval()
model.to(DEVICE)

# 检查权重文件是否存在
if not os.path.exists(CKPT_PATH):
    print(f"[WARNING] Checkpoint file {CKPT_PATH} not found!")
    print(f"[INFO] Please make sure the checkpoint file exists or update CKPT_PATH")
    print(f"[INFO] For vim_small, you might need to download the weights first")
    print(f"[INFO] You can set MODEL_TYPE = 'vim_tiny' to use the existing tiny model")
else:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    state_dict = model.state_dict()
    model.load_state_dict(ckpt, strict=True)
    print(f"[INFO] Successfully loaded checkpoint: {CKPT_PATH}")

# 3. 生成随机图片数据
all_imgs = torch.randn(NUM_IMAGES, 3, input_size, input_size)

# 4. FLOPs 统计（只需一次，取一张图即可）
# flop_inputs = all_imgs[0:1].to(DEVICE)
# flops = FlopCountAnalysis(model, flop_inputs)
# print(f"Single image FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

# 更精准的理论 FLOPs 估算（vim-tiny，224x224输入）
# img_size = 224
# patch_size = 16
# embed_dim = 192
# depth = 24
# d_state = 16
# num_classes = 1000
#
# num_patches = (img_size // patch_size) ** 2
#
# # Patch Embedding
# patch_embed_flops = 2 * embed_dim * num_patches * 3 * patch_size * patch_size
#
# # SelectiveScan/SSM
# ssm_flops_per_layer = 2 * embed_dim * num_patches * d_state
# ssm_total_flops = ssm_flops_per_layer * depth
#
# # MLP
# mlp_hidden_dim = 4 * embed_dim
# mlp_flops_per_layer = (
#     2 * num_patches * (embed_dim * mlp_hidden_dim) +  # 第一层全连接
#     num_patches * mlp_hidden_dim +                   # 激活
#     2 * num_patches * (mlp_hidden_dim * embed_dim)   # 第二层全连接
# )
# mlp_total_flops = mlp_flops_per_layer * depth
#
# # LayerNorm
# layernorm_flops_per_layer = 4 * num_patches * embed_dim
# layernorm_total_flops = layernorm_flops_per_layer * depth * 2  # 通常每层2次LayerNorm
#
# # Residual Add
# residual_flops_per_layer = num_patches * embed_dim
# residual_total_flops = residual_flops_per_layer * depth * 2  # 通常每层2次残差
#
# # Head
# head_flops = 2 * embed_dim * num_classes
#
# # 总 FLOPs
# total_flops = (
#     patch_embed_flops +
#     ssm_total_flops +
#     mlp_total_flops +
#     layernorm_total_flops +
#     residual_total_flops +
#     head_flops
# )
#
# print(f"[Precise Theoretical Estimate]")
# print(f"Patch Embedding FLOPs: {patch_embed_flops/1e6:.2f} MFLOPs")
# print(f"SelectiveScan FLOPs (total): {ssm_total_flops/1e6:.2f} MFLOPs")
# print(f"MLP FLOPs (total): {mlp_total_flops/1e6:.2f} MFLOPs")
# print(f"LayerNorm FLOPs (total): {layernorm_total_flops/1e6:.2f} MFLOPs")
# print(f"Residual Add FLOPs (total): {residual_total_flops/1e6:.2f} MFLOPs")
# print(f"Head FLOPs: {head_flops/1e6:.2f} MFLOPs")
# print(f"Total FLOPs (theoretical, single 224x224 image): {total_flops/1e9:.2f} GFLOPs\n")

# 准备输入数据
inputs = torch.randn(1, 3, input_size, input_size)

# 统计参数量、文件大小和推理时间
import time
import os
import numpy as np

def get_FileSize(filePath):
    filePath = str(filePath)
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)

# 计算模型文件大小
mb = get_FileSize(CKPT_PATH)

# 统计参数量
model_param_count = sum(p.numel() for p in model.parameters())

# 统计推理时间
start = time.time()
outputs = model(inputs)
infer_time_record = time.time() - start

print(f"📊 模型参数量: {model_param_count/1e6:.2f}M")
print(f"📊 模型文件大小: {mb}MB")
print(f"📊 单张图片推理时间: {np.mean(infer_time_record):.4f}秒")

# 1. 项目标准FLOPs计算方法（detectron2风格）
print("\n=== 项目标准FLOPs计算方法（detectron2风格）===")
print("🔧 这是项目det/tools/analyze_model.py中使用的标准FLOPs计算方法")
print("🔧 特点：基于fvcore的自动统计，标准化程度高")
try:
    # 添加det路径以导入项目标准分析工具
    det_path = os.path.join(os.path.dirname(__file__), 'det')
    if os.path.exists(det_path):
        sys.path.insert(0, det_path)
        print(f"[INFO] Added det path: {det_path}")
        
        from detectron2.utils.analysis import FlopCountAnalysis as DetFlopCountAnalysis
        from detectron2.utils.analysis import flop_count_operators
        
        # 方法1：使用detectron2的FlopCountAnalysis
        print("🔍 方法1：detectron2 FlopCountAnalysis")
        det_flops = DetFlopCountAnalysis(model, inputs)
        det_total_flops = det_flops.total()
        det_by_operator = det_flops.by_operator()
        
        print(f"detectron2 FLOPs: {det_total_flops/1e9:.3f}G")
        print("按操作类型统计:")
        for op, count in det_by_operator.items():
            print(f"  {op}: {count/1e9:.3f}G")
        
        # 方法2：使用flop_count_operators函数
        print("\n🔍 方法2：flop_count_operators")
        # 准备detectron2格式的输入
        det_inputs = [{"image": inputs}]
        op_flops = flop_count_operators(model, det_inputs)
        
        print("操作级别FLOPs统计:")
        for op, count in op_flops.items():
            print(f"  {op}: {count:.3f}G")
        
        sum_op_flops = sum(op_flops.values())
        print(f"总FLOPs (flop_count_operators): {sum_op_flops:.3f}G")
        
    else:
        print("❌ det目录不存在，跳过项目标准FLOPs计算")
        print("   请确保det目录在项目根目录下")
        
except ImportError as e:
    print(f"❌ 导入detectron2分析工具失败: {e}")
    print("   可能原因: detectron2未安装或路径配置问题")
except Exception as e:
    print(f"❌ 项目标准FLOPs计算失败: {e}")



# 2. 自定义理论FLOPs计算（针对BiMamba v2优化）
print("\n=== 自定义理论FLOPs计算（针对BiMamba v2优化）===")
print("🔧 特点：手动计算每个模块的FLOPs，包含BiMamba v2特有的双向处理操作")

# 双重验证：既从模型获取，又提供硬编码fallback

# 调试信息：显示模型的所有属性
print("🔍 调试信息：模型属性")
model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
print(f"模型属性数量: {len(model_attrs)}")
print(f"前10个属性: {model_attrs[:10]}")

# 查找可能包含层的属性
layer_attrs = [attr for attr in model_attrs if 'layer' in attr.lower() or 'block' in attr.lower()]
print(f"可能的层属性: {layer_attrs}")

try:
    # 方法1：从模型实例中获取参数
    img_size = model.patch_embed.img_size[0]
    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    num_classes = model.head.out_features
    num_patches = model.patch_embed.num_patches
    
    # 尝试获取depth - 使用更安全的方法
    depth = None
    if hasattr(model, 'blocks'):
        depth = len(model.blocks)
    elif hasattr(model, 'layers'):
        depth = len(model.layers)
    elif hasattr(model, 'depth'):
        depth = model.depth
    elif hasattr(model, 'num_layers'):
        depth = model.num_layers
    else:
        # 尝试从模型的其他属性推断
        for attr_name in dir(model):
            if 'layer' in attr_name.lower() or 'block' in attr_name.lower():
                attr = getattr(model, attr_name)
                if hasattr(attr, '__len__'):
                    depth = len(attr)
                    break
    
    if depth is None:
        depth = 24  # fallback
    
    # 尝试获取d_state - 使用更安全的方法
    d_state = 16  # 默认值
    try:
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            if hasattr(model.blocks[0], 'mixer') and hasattr(model.blocks[0].mixer, 'd_state'):
                d_state = model.blocks[0].mixer.d_state
    except:
        pass
    
    print("✅ 成功从模型获取参数")
    
except Exception as e:
    print(f"⚠️ 从模型获取参数失败: {e}")
    print(f"🔄 使用硬编码参数（{MODEL_TYPE} 标准配置）")
    
    # 方法2：硬编码参数（根据模型类型）
    img_size = 224
    patch_size = 16
    embed_dim = EMBED_DIM  # 使用全局变量
    depth = DEPTH  # 使用全局变量
    d_state = D_STATE  # 使用全局变量
    num_classes = 1000
    num_patches = (img_size // patch_size) ** 2

# 验证参数合理性
print(f"📊 模型参数验证:")
print(f"   img_size: {img_size}")
print(f"   patch_size: {patch_size}")
print(f"   embed_dim: {embed_dim}")
print(f"   depth: {depth}")
print(f"   d_state: {d_state}")
print(f"   num_classes: {num_classes}")
print(f"   num_patches: {num_patches}")

# 参数合理性检查
assert img_size == 224, f"img_size 应该是 224，实际是 {img_size}"
assert patch_size == 16, f"patch_size 应该是 16，实际是 {patch_size}"
assert embed_dim == EMBED_DIM, f"embed_dim 应该是 {EMBED_DIM}，实际是 {embed_dim}"
assert depth == DEPTH, f"depth 应该是 {DEPTH}，实际是 {depth}"
assert d_state == D_STATE, f"d_state 应该是 {D_STATE}，实际是 {d_state}"
assert num_patches == 196, f"num_patches 应该是 196，实际是 {num_patches}"

print("✅ 所有参数验证通过！")

# 1. Patch Embedding FLOPs (thop 能统计到)
patch_embed_flops = 2 * embed_dim * num_patches * 3 * patch_size * patch_size
print(f"Patch Embedding FLOPs: {patch_embed_flops/1e6:.2f} MFLOPs")

# 2. Mamba 块 FLOPs (thop 统计不到)
# 每个 Mamba 块包含：
# - 因果卷积
# - Selective Scan (SSM) - 注意：BiMamba v2 有前向和反向两个处理
# - MLP
# - LayerNorm

# 因果卷积 FLOPs (BiMamba v2 有前向和反向两个卷积)
conv1d_flops_per_layer = 2 * embed_dim * num_patches * 4  # width=4, 双向处理
conv1d_total_flops = conv1d_flops_per_layer * depth

# Selective Scan FLOPs (BiMamba v2 有前向和反向两个 SSM)
ssm_flops_per_layer = 2 * embed_dim * num_patches * d_state * 2  # 双向处理，所以乘以2
ssm_total_flops = ssm_flops_per_layer * depth

# MLP FLOPs (BiMamba v2 的实际结构)
# 从源码分析：BiMamba v2 有前向和反向两个处理路径
# 每个路径包含：x_proj, dt_proj, out_proj

# 输入投影 (x_proj 和 x_proj_b) - 双向处理
# x_proj: (batch, seqlen, embed_dim) -> (batch*seqlen, embed_dim*2)
x_proj_flops_per_layer = 2 * num_patches * embed_dim * (embed_dim * 2)  # 双向处理，expand=2
x_proj_total_flops = x_proj_flops_per_layer * depth

# 时间步投影 (dt_proj 和 dt_proj_b) - 双向处理
# dt_proj: (batch*seqlen, embed_dim*2) -> (batch*seqlen, dt_rank)
dt_rank = 16  # 默认值
dt_proj_flops_per_layer = 2 * num_patches * (embed_dim * 2) * dt_rank  # 双向处理
dt_proj_total_flops = dt_proj_flops_per_layer * depth

# 输出投影 (out_proj) - 只在最后统一处理，不是每层都有
out_proj_flops_per_layer = 2 * num_patches * embed_dim * embed_dim  # 双向处理
out_proj_total_flops = out_proj_flops_per_layer * depth  # 每层都有输出投影

# 总 MLP FLOPs (包含每层的输入投影、时间步投影，以及每层的输出投影)
mlp_total_flops = x_proj_total_flops + dt_proj_total_flops + out_proj_total_flops

# LayerNorm FLOPs
layernorm_flops_per_layer = 4 * num_patches * embed_dim
layernorm_total_flops = layernorm_flops_per_layer * depth * 2  # 通常每层2次LayerNorm

# 残差连接 FLOPs
residual_flops_per_layer = num_patches * embed_dim
residual_total_flops = residual_flops_per_layer * depth * 2

# BiMamba v2 特有的额外计算
# 1. 序列翻转操作
flip_flops_per_layer = num_patches * embed_dim  # 翻转操作
flip_total_flops = flip_flops_per_layer * depth * 2  # 前向和反向各一次

# 2. 输出合并操作
merge_flops_per_layer = num_patches * embed_dim  # 加法操作
merge_total_flops = merge_flops_per_layer * depth

# 3. 除法操作 (if_divide_out=True)
divide_flops_per_layer = num_patches * embed_dim  # 除法操作
divide_total_flops = divide_flops_per_layer * depth

# 3. Head FLOPs (thop 能统计到)
head_flops = 2 * embed_dim * num_classes

# 总理论 FLOPs
total_theoretical_flops = (
    patch_embed_flops +
    conv1d_total_flops +
    ssm_total_flops +
    mlp_total_flops +
    layernorm_total_flops +
    residual_total_flops +
    flip_total_flops +      # BiMamba v2 翻转操作
    merge_total_flops +     # BiMamba v2 合并操作
    divide_total_flops +    # BiMamba v2 除法操作
    head_flops
)

print(f"📊 FLOPs 分解 (理论计算):")
print(f"  Patch Embedding: {patch_embed_flops/1e9:.3f}G")
print(f"  因果卷积 (BiMamba v2): {conv1d_total_flops/1e9:.3f}G")
print(f"  Selective Scan (BiMamba v2): {ssm_total_flops/1e9:.3f}G")
print(f"  输入投影 (BiMamba v2): {x_proj_total_flops/1e9:.3f}G")
print(f"  时间步投影 (BiMamba v2): {dt_proj_total_flops/1e9:.3f}G")
print(f"  输出投影 (BiMamba v2): {out_proj_total_flops/1e9:.3f}G")
print(f"  LayerNorm: {layernorm_total_flops/1e9:.3f}G")
print(f"  残差连接: {residual_total_flops/1e9:.3f}G")
print(f"  序列翻转 (BiMamba v2): {flip_total_flops/1e9:.3f}G")
print(f"  输出合并 (BiMamba v2): {merge_total_flops/1e9:.3f}G")
print(f"  除法操作 (BiMamba v2): {divide_total_flops/1e9:.3f}G")
print(f"  Head: {head_flops/1e9:.3f}G")
print(f"  总计: {total_theoretical_flops/1e9:.3f}G")



# 3. 综合FLOPs对比总结
print(f"\n=== 综合FLOPs对比总结 ===")
print(f"📊 {MODEL_TYPE} 模型 FLOPs 统计对比:")

# 尝试获取项目标准FLOPs结果进行对比
try:
    if 'det_total_flops' in locals():
        print(f"  项目标准 (detectron2): {det_total_flops/1e9:.3f}G")
    if 'sum_op_flops' in locals():
        print(f"  项目标准 (operators): {sum_op_flops:.3f}G")
except:
    pass

print(f"  自定义理论计算:      {total_theoretical_flops/1e9:.3f}G")

print(f"\n💡 说明:")
print(f"  - 项目标准：detectron2框架的标准FLOPs计算方法")
print(f"  - 自定义理论：基于模型架构的手动计算，包含BiMamba v2的特殊操作")

# 4. 吞吐量统计
# num_batches = (len(all_imgs) + BATCH_SIZE - 1) // BATCH_SIZE
# start = time.time()
# with torch.no_grad():
#     for i in range(num_batches):
#         batch = all_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(DEVICE)
#         _ = model(batch)
# end = time.time()
# total_time = end - start
# throughput = len(all_imgs) / total_time

# print(f"Processed {len(all_imgs)} images in {total_time:.2f} seconds")
# print(f"Average throughput: {throughput:.2f} images/second")
# print(f"Average time per image: {total_time / len(all_imgs):.4f} seconds")

# 吞吐量统计参数
batch_size = 1
num_images = 1000
log_interval = 10
log_file = "throughput_log.txt"

throughputs = []
start_time = time.time()

with open(log_file, "w") as f:
    for i in range(num_images):
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        t0 = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        t1 = time.time()
        throughput = batch_size / (t1 - t0)
        throughputs.append(throughput)
        
        if (i + 1) % log_interval == 0:
            avg_throughput = sum(throughputs[-log_interval:]) / log_interval
            f.write(f"Batch {i+1}: {avg_throughput:.2f} images/sec\n")
            print(f"Batch {i+1}: {avg_throughput:.2f} images/sec")

end_time = time.time()
total_time = end_time - start_time
avg_throughput = sum(throughputs) / len(throughputs)

print(f"\n=== 吞吐量统计结果 ===")
print(f"总处理时间: {total_time:.2f} 秒")
print(f"平均吞吐量: {avg_throughput:.2f} images/sec")
print(f"平均每张图片时间: {total_time / num_images:.4f} 秒")
print(f"详细日志已保存到: {log_file}")

# 4. 统计参数量
model_param_count = sum(p.numel() for p in model.parameters())
print(f"\n📊 模型参数量: {model_param_count/1e6:.2f}M")

# 保存模型权重到本地
model_cpu = model.to('cpu')
save_path = f"{MODEL_TYPE}_cpu.pth"
torch.save(model_cpu.state_dict(), save_path)
print(f"模型权重已保存到 {save_path}") 