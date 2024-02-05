import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_clip):
        super(TextEncoder, self).__init__()
        self.model_clip = model_clip

    def forward(self, inputs):
        outputs = self.model_clip.encode_text(inputs)
        text_features = outputs / outputs.norm(dim=-1, keepdim=True)
        return text_features

class VisualEncoder(nn.Module):
    def __init__(self, model_clip):
        super(VisualEncoder, self).__init__()
        self.model_clip = model_clip

    def forward(self, inputs):
        outputs = self.model_clip.encode_image(inputs)
        img_features = outputs / outputs.norm(dim=-1, keepdim=True)
        return img_features
