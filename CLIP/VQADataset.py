import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, data, cls_to_idx, img_preprocess, text_tokenizer, device, root_dir="./data/val2014-resised"):
        self.data = data
        self.cls_to_idx = cls_to_idx
        self.img_preprocess = img_preprocess
        self.text_tokennizer = text_tokenizer
        self.device = device
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx]["image_path"])
        image = Image.open(img_path).convert("RGB")
        question = self.data[idx]["question"]
        label = self.cls_to_idx[self.data[idx]["answer"]]

        # Remove the unsqueezing for the image tensor
        if self.img_preprocess is not None:
            image = self.img_preprocess(image).to(self.device)

        if self.text_tokennizer is not None:
            question = self.text_tokennizer(question).squeeze(0).to(self.device)

        label = torch.tensor(label, dtype=torch.long).to(self.device)

        sample = {
            "image": image,
            "question": question,
            "label": label
        }

        return sample
