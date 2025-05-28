import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Specify the GPU you want to use

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import umap
import matplotlib.pyplot as plt

import numpy as np

attn_maps = []

def hook_fn_attention(module, input, output):
    # output is after softmax in Dinov2SelfAttention
    hidden_states = input[0]  # shape: [B, N, C]
    B, N, C = hidden_states.shape

    # Project to Q, K, V
    q = module.query(hidden_states)
    k = module.key(hidden_states)

    # Reshape to (B, heads, tokens, head_dim)
    q = q.view(B, N, module.num_attention_heads, module.attention_head_size).permute(0, 2, 1, 3)
    k = k.view(B, N, module.num_attention_heads, module.attention_head_size).permute(0, 2, 1, 3)

    q = q.detach().cpu()
    k = k.detach().cpu()

    attn_logits = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)  # [B, heads, tokens, tokens]
    attn_probs = torch.nn.functional.softmax(attn_logits, dim=-1)

    attn_maps.append(attn_probs)

def plot_cls_attention(attn_maps):
    """
    attn_maps: List of tensors of shape [heads, tokens, tokens], one per layer.
               e.g., [layer0_attn, layer1_attn, ..., layerN_attn]
               Each tensor is [6, 257, 257] for DINOv2-small.
    """
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]  # 6 heads

    fig, axs = plt.subplots(num_layers, num_heads, figsize=(2 * num_heads, 2 * num_layers))

    for layer_idx, attn in enumerate(attn_maps):
        # attention from CLS token to all other tokens (excluding self)
        attn = attn[0]
        cls_attn = attn[:, 0, 1:]  # shape: [heads, 256]

        for head_idx in range(num_heads):
            attn_map = cls_attn[head_idx].reshape(16, 16).detach().cpu()  # [16x16] patch map
            ax = axs[layer_idx, head_idx] if num_layers > 1 else axs[head_idx]

            im = ax.imshow(attn_map, cmap='viridis')
            ax.axis('off')
            if layer_idx == 0:
                ax.set_title(f"H{head_idx}", fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=8)

    plt.suptitle("CLS → Patch Attention\n(L=Layer, H=Head)", fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = "https://www.hartz.com/wp-content/uploads/2022/04/small-dog-owners-1.jpg"
    # url = "https://news.harvard.edu/wp-content/uploads/2023/11/dog_brains_2500.png?resize=1024,576"
    # url = "https://i.guim.co.uk/img/media/ba55356947303677d52fb6f6239d033fb17f1aab/0_0_2539_2032/master/2539.jpg?width=965&dpr=1&s=none&crop=none"
    # url = "https://hips.hearstapps.com/hmg-prod/images/small-dogs-6626cf74dfe17.jpg?crop=1xw:0.84375xh;center,top&resize=1200:*"
    url = "https://hips.hearstapps.com/hmg-prod/images/woman-with-dog-outside-1558449251.jpg?crop=0.680xw:1.00xh;0,0&resize=980:*"
    # url =  "https://www.qimacros.com/excel-charts-qimacros/column-chart-excel.png"
#
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert("RGB")


    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=False)
    # model = AutoModel.from_pretrained('facebook/dinov2-base')

    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-base')
    # model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast = False)  
    model = AutoModel.from_pretrained('facebook/dinov2-small')


    height, weight =  processor.crop_size["height"], processor.crop_size['width']
    print(processor)
    print(model)

    hooks = []
    for layer in model.encoder.layer:
        attn_module = layer.attention.attention  # Dinov2SelfAttention
        hook = attn_module.register_forward_hook(hook_fn_attention)
        hooks.append(hook)

    
    # target_layer = model.encoder.layer[-1].attention.attention
    # hook = target_layer.register_forward_hook(hook_fn_attention)


    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
        features = outputs.last_hidden_state

        features = features.detach().squeeze(0)

        # take data unless CLS
        flat = features[1:].numpy()

    # registers = features[257:261].numpy()  # shape: (4, 768)

    for hook in hooks:
        hook.remove()

    # Dimensionality reduction
    reduced = umap.UMAP(n_components=3).fit_transform(flat)
    reduced_img = reduced.reshape(16, 16, 3)
  
    # Normalize to 0-255
    rgb = ((reduced_img - reduced_img.min()) / reduced_img.ptp() * 255).astype(np.uint8)

    color_img = Image.fromarray(rgb, mode='RGB')
    upscaled = color_img.resize((height, weight ), resample=Image.BICUBIC)

    resized_image = image.resize((height, weight ), resample=Image.BICUBIC)

    blended = Image.blend(resized_image, upscaled, alpha=0.6)  # 50% overlap
    blended.show()


    plot_cls_attention(attn_maps)




if __name__ == "__main__":
    main()