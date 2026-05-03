import torch
import torch.nn as nn
from torchvision import transforms
from huggingface_hub import hf_hub_download
import streamlit as st
from PIL import Image

# ✅ LOAD MODEL FROM HUGGINGFACE
@st.cache_resource
def load_ct_model():
    model_path = hf_hub_download(
        repo_id="the-kshitij-chauhan/brain-tumour-model",
        filename="ct_classification_model.pth"
    )

    # 🔥 EXACT SAME ARCHITECTURE AS TRAINING (NO CLASS WRAPPER)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


# Load once
ct_model = load_ct_model()


# ✅ PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ✅ CLASS LABELS (must match training order)
classes = ['aneurysm', 'cancer', 'tumor']


# ✅ MAIN PREDICT FUNCTION
def predict(image_file):

    # Load image
    image = Image.open(image_file).convert("RGB")

    # Preprocess
    image = transform(image).unsqueeze(0)

    # Inference
    with torch.inference_mode():
        outputs = ct_model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]