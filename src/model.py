"""
CRNN + CTC model za OCR srpskih računa
Arhitektura: ResNet CNN -> BiLSTM -> CTC
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Osnovni rezidualni blok za CNN deo."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=48, hidden_size=128, num_lstm_layers=1):
        super().__init__()

        self.cnn = nn.Sequential(
            # Blok 1: 1 -> 32 | H: 48 -> 24
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Blok 2: 32 -> 64 | H: 24 -> 12
            ResidualBlock(32, 64),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 3: 64 -> 128 | H: 12 -> 6
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 4: 128 -> 256 | H: 6 -> 3
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 5: 256 -> 512 | H: 3 -> 1
            ResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),

            nn.Dropout2d(0.3),  # bilo 0.2
        )

        self.dropout = nn.Dropout(p=0.4)  # novo — između CNN i LSTM

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=False,
            bidirectional=True,
            dropout=0.0,  # 0 jer je num_lstm_layers=1 (dropout ne radi sa 1 slojem)
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)               # [B, 512, 1, W/2]
        features = features.squeeze(2)       # [B, 512, W/2]
        features = features.permute(2, 0, 1) # [T, B, 512]

        features = self.dropout(features)    # novo — regularizacija pre LSTM-a

        lstm_out, _ = self.lstm(features)    # [T, B, hidden_size * 2]
        logits = self.fc(lstm_out)           # [T, B, num_classes]

        return logits