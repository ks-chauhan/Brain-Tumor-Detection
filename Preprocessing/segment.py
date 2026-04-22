from PIL import Image
from torchvision import transforms

transform_seg = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def preprocess_seg(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    image = image.convert("L")
    image = transform_seg(image)
    image = image.unsqueeze(0)

    return image
