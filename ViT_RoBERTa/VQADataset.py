import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(
        self,
        data,
        cls_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        root_dir="./data/val2014-resised"
    ):

        self.data = data
        self.cls_to_idx = cls_to_idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx
    ):

        img_path = os.path.join(self.root_dir, self.data[idx]["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.img_feature_extractor:
            img = self.img_feature_extractor(image, return_tensors="pt")
            img = {k: v.to(self.device).squeeze(0) for k, v in img.items()}
        else:
            img = image

        question = self.data[idx]["question"]
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=20,
                return_tensors="pt"
            )
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}

        label = self.cls_to_idx[self.data[idx]["answer"]]
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        sample = {
            "image": img,
            "question": question,
            "label": label
        }

        return sample
