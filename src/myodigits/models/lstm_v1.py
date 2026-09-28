import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=20, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, # warning
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):               # (B, 8, 400)
        x = x.permute(0, 2, 1)        # -> (B, 400, 8)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (B, hidden_size)
        return self.head(last_hidden)  # (B, 20)