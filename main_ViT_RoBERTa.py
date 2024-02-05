
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from transformers import AutoTokenizer

from src.train import fit
from src.load_data import load_data
from ViT_RoBERTa.VQADataset import VQADataset
from ViT_RoBERTa.classifier import Classifier
from ViT_RoBERTa.encoder import TextEncoder, VisualEncoder
from ViT_RoBERTa.VQAModel import VQAModel
from src.evaluate import evaluate
from src.config import get_config

def main():

    config = get_config()

    data_path = config.data_path
    train_data = load_data(config.train_data_path)
    val_data = load_data(config.val_data_path)
    test_data = load_data(config.test_data_path)
    img_feature_extractor = ViTImageProcessor.from_pretrained(config.img_feature_extractor_name)
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer_name)
    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    lr = config.lr
    epochs = config.epochs
    scheduler_step_size = config.scheduler_step_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    classes = set([sample["answer"] for sample in train_data])

    # Dictionary mapping classes
    cls_to_idx = {
        cls_name: idx for idx, cls_name in enumerate(classes)
    }

    idx_to_cls = {
        idx: cls_name for idx, cls_name in enumerate(classes)
    }

    train_dataset = VQADataset(
        train_data,
        cls_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        data_path
    )

    val_dataset = VQADataset(
        val_data,
        cls_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        data_path
    )

    test_dataset = VQADataset(
        test_data,
        cls_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        data_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    text_encoder = TextEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    classifier = Classifier().to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(device)

    model.freeze()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=0.1
    )

    train_losses, val_losses, model = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs
    )

    torch.save(model.state_dict(), 'test_model.pth')
    print("Model saved successfully.")

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f'Test loss: {test_loss:.4f} Test Acc:{test_acc:.4f}')

if __name__ == '__main__':
    main()
