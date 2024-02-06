import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CNN_LSTM.train import fit
from CNN_LSTM.tokenize import get_tokens
from src.load_data import load_data
from CNN_LSTM.VQADataset import VQADataset
from CNN_LSTM.VQAModel import VQAModel
from src.evaluate import evaluate
from src.config import get_config
from torchvision import transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

def main():

    config = get_config()

    data_path = config.data_path
    train_data = load_data(config.train_data_path)
    val_data = load_data(config.val_data_path)
    test_data = load_data(config.test_data_path)
    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    img_model_name = config.cnn_model_name
    lr = config.lr
    epochs = config.epochs
    scheduler_step_size = config.scheduler_step_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    vocab = build_vocab_from_iterator(
    get_tokens(train_data),
    min_freq=2 ,
    specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first = True
    )
    vocab.set_default_index(vocab['<unk>'])

    classes = set([sample["answer"] for sample in train_data])

    # Dictionary mapping classes
    cls_to_idx = {
        cls_name: idx for idx, cls_name in enumerate(classes)
    }

    idx_to_cls = {
        idx: cls_name for idx, cls_name in enumerate(classes)
    }

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)),
    ])

    train_dataset = VQADataset(
        train_data,
        cls_to_idx=cls_to_idx,
        transform=transform,
        vocab=vocab,
        root_dir=data_path
    )

    val_dataset = VQADataset(
        val_data,
        cls_to_idx=cls_to_idx,
        transform=transform,
        vocab=vocab,
        root_dir=data_path
    )

    test_dataset = VQADataset(
        test_data,
        cls_to_idx=cls_to_idx,
        transform=transform,
        vocab=vocab,
        root_dir=data_path
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

    model=VQAModel(
            n_classes=2,
            img_model_name=img_model_name,
            n_layers=2,
            embedding_dim=128,
            hidden_size=128,
            dropout=0.2,
            vocab=vocab
    ).to(device)

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

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully.")

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f'Test loss: {test_loss:.4f} Test Acc:{test_acc:.4f}')

if __name__ == '__main__':
    main()
