
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,
                 input_size=768*2,
                 hidden_size=512,
                 n_classes=2,
                 n_layers=1,
                 dropout=0.2):

        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout=nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
