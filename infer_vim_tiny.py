import torch
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis
import time
import sys
import os

# 1. 导入模型注册函数
sys.path.append(os.path.dirname(__file__))  # 保证能import本地模块
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

# 配置
CKPT_PATH = 'vim_t_midclstok_76p1acc.pth'  # 你的权重路径
DEVICE = 'cpu'  # 树莓派建议用cpu
BATCH_SIZE = 8
NUM_IMAGES = 100  # 可改为1000
input_size = 224

# 2. 加载模型
model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(num_classes=1000)
model.eval()
model.to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
if 'model' in ckpt:
    ckpt = ckpt['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    if k in ckpt and k in state_dict and ckpt[k].shape != state_dict[k].shape:
        del ckpt[k]
if 'pos_embed' in ckpt:
    pos_embed_checkpoint = ckpt['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    ckpt['pos_embed'] = new_pos_embed
model.load_state_dict(ckpt, strict=False)

# 3. 生成随机图片数据
all_imgs = torch.randn(NUM_IMAGES, 3, input_size, input_size)

# 4. FLOPs 统计（只需一次，取一张图即可）
# flop_inputs = all_imgs[0:1].to(DEVICE)
# flops = FlopCountAnalysis(model, flop_inputs)
# print(f"Single image FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

# 更精准的理论 FLOPs 估算（vim-tiny，224x224输入）
img_size = 224
patch_size = 16
embed_dim = 192
depth = 24
d_state = 16
num_classes = 1000

num_patches = (img_size // patch_size) ** 2

# Patch Embedding
patch_embed_flops = 2 * embed_dim * num_patches * 3 * patch_size * patch_size

# SelectiveScan/SSM
ssm_flops_per_layer = 2 * embed_dim * num_patches * d_state
ssm_total_flops = ssm_flops_per_layer * depth

# MLP
mlp_hidden_dim = 4 * embed_dim
mlp_flops_per_layer = (
    2 * num_patches * (embed_dim * mlp_hidden_dim) +  # 第一层全连接
    num_patches * mlp_hidden_dim +                   # 激活
    2 * num_patches * (mlp_hidden_dim * embed_dim)   # 第二层全连接
)
mlp_total_flops = mlp_flops_per_layer * depth

# LayerNorm
layernorm_flops_per_layer = 4 * num_patches * embed_dim
layernorm_total_flops = layernorm_flops_per_layer * depth * 2  # 通常每层2次LayerNorm

# Residual Add
residual_flops_per_layer = num_patches * embed_dim
residual_total_flops = residual_flops_per_layer * depth * 2  # 通常每层2次残差

# Head
head_flops = 2 * embed_dim * num_classes

# 总 FLOPs
total_flops = (
    patch_embed_flops +
    ssm_total_flops +
    mlp_total_flops +
    layernorm_total_flops +
    residual_total_flops +
    head_flops
)

print(f"[Precise Theoretical Estimate]")
print(f"Patch Embedding FLOPs: {patch_embed_flops/1e6:.2f} MFLOPs")
print(f"SelectiveScan FLOPs (total): {ssm_total_flops/1e6:.2f} MFLOPs")
print(f"MLP FLOPs (total): {mlp_total_flops/1e6:.2f} MFLOPs")
print(f"LayerNorm FLOPs (total): {layernorm_total_flops/1e6:.2f} MFLOPs")
print(f"Residual Add FLOPs (total): {residual_total_flops/1e6:.2f} MFLOPs")
print(f"Head FLOPs: {head_flops/1e6:.2f} MFLOPs")
print(f"Total FLOPs (theoretical, single 224x224 image): {total_flops/1e9:.2f} GFLOPs\n")

# 5. 吞吐量统计
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
        elapsed = t1 - t0
        throughput = batch_size / elapsed if elapsed > 0 else 0
        throughputs.append(throughput)
        if (i + 1) % log_interval == 0:
            avg_throughput = sum(throughputs[-log_interval:]) / log_interval
            msg = f"Image {i+1}/{num_images}, Avg throughput (last {log_interval}): {avg_throughput:.4f} img/s"
            print(msg)
            f.write(msg + "\n")
    total_time = time.time() - start_time
    overall_avg = sum(throughputs) / len(throughputs)
    summary = f"\nTotal time: {total_time:.2f}s, Overall avg throughput: {overall_avg:.2f} img/s"
    print(summary)
    f.write(summary + "\n")

# 统计参数量
model_param_count = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {model_param_count/1e6:.2f}M")

# 保存模型权重到本地
model_cpu = model.to('cpu')
torch.save(model_cpu.state_dict(), "vim_tiny_cpu.pth")
print("模型权重已保存到 vim_tiny_cpu.pth") 