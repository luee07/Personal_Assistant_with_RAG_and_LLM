import torch

from transformers import (
    CLIPProcessor,
    CLIPModel
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        features = model.get_image_features(
            **inputs
        )

    return features

def get_text_embedding(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        features = model.get_text_features(
            **inputs
        )

    return features