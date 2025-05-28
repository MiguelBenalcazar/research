import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Specify the GPU you want to use

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import umap

import numpy as np


def main():
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = "https://www.hartz.com/wp-content/uploads/2022/04/small-dog-owners-1.jpg"
    # url = "https://news.harvard.edu/wp-content/uploads/2023/11/dog_brains_2500.png?resize=1024,576"
    # url = "https://i.guim.co.uk/img/media/ba55356947303677d52fb6f6239d033fb17f1aab/0_0_2539_2032/master/2539.jpg?width=965&dpr=1&s=none&crop=none"
    # url = "https://hips.hearstapps.com/hmg-prod/images/small-dogs-6626cf74dfe17.jpg?crop=1xw:0.84375xh;center,top&resize=1200:*"
    # url = "https://hips.hearstapps.com/hmg-prod/images/woman-with-dog-outside-1558449251.jpg?crop=0.680xw:1.00xh;0,0&resize=980:*"
    url =  "https://www.qimacros.com/excel-charts-qimacros/column-chart-excel.png"

    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert("RGB")


    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=False)
    # model = AutoModel.from_pretrained('facebook/dinov2-base')

    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-base')
    # model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast = False)
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    
    print(processor)
    print(model)



    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        print( "Pooler",    outputs.pooler_output.squeeze(0).tolist())
    
        features = outputs.last_hidden_state
        features = features.detach()
        features = features.squeeze(0)

       
        # take data unless CLS
        # flat = features[1:257].numpy()
        flat =  features[1:].numpy() 
        print(f"Shape of features = {flat.shape}")

    print("CLS DATA ----------------------")
    print(features[0,:].size())

    # registers = features[257:261].numpy()  # shape: (4, 768)


    # Dimensionality reduction
    reduced = umap.UMAP(n_components=3).fit_transform(flat)
    reduced_img = reduced.reshape(16, 16, 3)
  
    # Normalize to 0-255
    rgb = ((reduced_img - reduced_img.min()) / reduced_img.ptp() * 255).astype(np.uint8)

    color_img = Image.fromarray(rgb, mode='RGB')
    upscaled = color_img.resize(image.size, resample=Image.BICUBIC)

    blended = Image.blend(image, upscaled, alpha=0.8)  # 50% overlap
    blended.show()



if __name__ == "__main__":
    main()