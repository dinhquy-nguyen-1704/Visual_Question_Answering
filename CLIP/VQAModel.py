import torch
import torch.nn as nn

class VQAModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        text_tokenizer,
        classifier
    ):
        super(VQAModel, self).__init__()
        self.visual_encoder = visual_encoder
        self.text_tokenizer = text_tokenizer
        self.classifier = classifier

    def forward(
        self,
        image,
        question
    ):
        img_out = self.visual_encoder(image)
        text_out = self.text_tokenizer(question)
        x = torch.cat((text_out, img_out), dim=1)
        out = self.classifier(x)

        return out

    def freeze(
        self,
        visual_encoder=True,
        text_tokenizer=True,
        classifier=False
    ):

        if visual_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        if text_tokenizer:
            for param in self.text_tokenizer.parameters():
                param.requires_grad = False

        if classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
