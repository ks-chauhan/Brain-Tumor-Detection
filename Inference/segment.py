from huggingface_hub import hf_hub_download
import torch
import streamlit as st
from Preprocessing.segment import preprocess_seg
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = block(1,64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = block(128,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = block(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = block(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2,e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1,e1], dim=1))

        return self.out(d1)

@st.cache_resource
def load_seg_model():
    model_path = hf_hub_download(
        repo_id="the-kshitij-chauhan/brain-tumour-model",
        filename="unet_segmentation.pth" 
    )

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model

def mask_to_numpy(mask):
    return mask.squeeze().cpu().numpy()

def overlay_mask(image_file, mask):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128,128))
    image = np.array(image)

    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    colored_mask = np.zeros_like(image)
    colored_mask[:,:,0] = mask  # red

    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    return overlay

seg_model = load_seg_model()


def predict_mask(image_file):

    original_image = image_file
    image = preprocess_seg(image_file)

    with torch.inference_mode():
        output = seg_model(image)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float()

    overlay = overlay_mask(original_image, mask)
    mask_np = mask.squeeze().cpu().numpy()

    return overlay, mask_np