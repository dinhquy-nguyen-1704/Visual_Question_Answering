import torch
import torch.nn as nn
import timm

class VQAModel(nn.Module):
    def __init__(
            self,
            n_classes,
            img_model_name='resnet50',
            n_layers=2,
            embedding_dim=300,
            hidden_size=128,
            dropout=0.2,
            vocab=None
    ):

        super(VQAModel, self).__init__()
        self.img_encoder=timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=hidden_size
        )

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.layernorm = nn.LayerNorm(hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*3, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, img, text):
        img_features = self.img_encoder(img)

        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm(text_emb)

        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.layernorm(lstm_out)

        combined = torch.cat((img_features, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
