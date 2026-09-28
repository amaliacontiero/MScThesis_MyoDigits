import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 20)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)
        x = self.head(x)
        return x