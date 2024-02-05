import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from CNN_LSTM.tokenize import get_tokens, tokenize

class VQADataset(Dataset):
    def __init__(
            self,
            data,
            cls_to_idx,
            max_seq_len=30,
            transform=None,
            vocab=None,
            root_dir="./val2014-resised"
    ):

        self.transform = transform
        self.data = data
        self.max_seq_len = max_seq_len
        self.cls_to_idx = cls_to_idx
        self.vocab=vocab
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data[index]["image_path"])

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        question = self.data[index]["question"]
        question = tokenize(question, self.max_seq_len, self.vocab)
        question = torch.tensor(question, dtype=torch.long)

        label = self.data[index]["answer"]
        label = self.cls_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)

        return img, question, label
