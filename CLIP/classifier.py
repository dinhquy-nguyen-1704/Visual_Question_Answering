import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,
                 input_size=512*2,
                 hidden_size=512,
                 n_classes=2,
                 n_layers=1,
                 dropout=0.2):

        super(Classifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # Chuyển đầu vào qua mạng MLP
        x = self.mlp(x)
        return x
