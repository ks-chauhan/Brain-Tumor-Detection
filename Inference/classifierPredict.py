import torch
import torch.nn as nn
from Preprocessing.classifier import preprocess
import torchvision.models as models
from huggingface_hub import hf_hub_download
import streamlit as st

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="the-kshitij-chauhan/brain-tumour-model",
        filename="brain_tumor_resnet50.pth"
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model

model = load_model()

def predict(image):
    image = preprocess(image)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    with torch.inference_mode():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    
    return classes[pred]