import torch

def cosine_similarity(
        image_embedding,
        text_embedding):

    similarity = torch.nn.functional.cosine_similarity(
        image_embedding,
        text_embedding
    )

    return similarity.item()