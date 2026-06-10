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
    """
    CRNN arhitektura:
    - CNN (ResNet-like) za ekstrakciju features (širina se smanjuje samo 2 puta!)
    - BiLSTM za sekvencijalno modelovanje
    - Linearni sloj za predikciju karaktera
    """
    def __init__(self, num_classes, img_height=48, hidden_size=256, num_lstm_layers=2):
        super().__init__()

        # ── CNN DEO (Smanjenje širine za samo 2x kroz celu mrežu) ─────────────────
        self.cnn = nn.Sequential(
            # Blok 1: 1 -> 32 | H: 48 -> 24 | W: W -> W/2
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Blok 2: 32 -> 64 | H: 24 -> 12 | W: W/2 ostaje W/2 (asimetrični pooling)
            ResidualBlock(32, 64),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 3: 64 -> 128 | H: 12 -> 6 | W: W/2 ostaje W/2
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 4: 128 -> 256 | H: 6 -> 3 | W: W/2 ostaje W/2
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 5: 256 -> 512 | H: 3 -> 1 | W: W/2 ostaje W/2
            ResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),

            nn.Dropout2d(0.2),
        )

        lstm_input_size = 512

        # ── BiLSTM DEO ───────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=False,   # Izlaz [T, B, features]
            bidirectional=True,  # Čita tekst u oba smera
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
        )

        # ── IZLAZNI SLOJ ─────────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 zbog dvosmernog LSTM-a

    def forward(self, x):
        # x: [B, 1, H, W]
        features = self.cnn(x)               # Izlaz: [B, 512, 1, W/2]
        features = features.squeeze(2)       # Izlaz: [B, 512, W/2]
        features = features.permute(2, 0, 1) # Izlaz: [W/2, B, 512] -> [T, B, features]

        lstm_out, _ = self.lstm(features)    # Izlaz: [T, B, hidden_size * 2]
        logits = self.fc(lstm_out)           # Izlaz: [T, B, num_classes]

        return logits