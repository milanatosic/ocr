"""
CRNN + CTC model za OCR srpskih racuna
Arhitektura: ResNet CNN -> BiLSTM -> CTC
"""

import torch
import torch.nn as nn

# Rezidualni blok je osnovna jedinica ResNet arhitekture 
# Problem dubokih mreza je sto gradijent nestaje dok se propagira nazad kroz slojeve - mreza prestaje da uci
# Resenje je precica - ulaz se dodaje direktno na izlaz
class ResidualBlock(nn.Module):
    """Osnovni rezidualni blok za CNN deo."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False) #prvi konvolucijski sloj
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False) #drugi konvolucijski sloj
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() #precica
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
    - CNN (ResNet-like) za ekstrakciju features
    - BiLSTM za sekvencijalno modelovanje
    - Linearni sloj za predikciju karaktera
    """
    def __init__(self, num_classes, img_height=48, hidden_size=256, num_lstm_layers=2):
        super().__init__()

        # ── CNN deo ──────────────────────────────────────────────────────────
        """
        CNN prima ulaz oblika [B, 1, 48, W], gde je B - batch size, 1 je jedan kanal(greyscale), 48 je visina, W je sirina
        Svaki sloj smanjuje visinu poligonom dok ne dobijemo visinu 1 - sva informacija o visini je sabita u 512 feature mapa
        Sirina se smanjuje mnogo manje jer koristimo asimetrican pooling(2, 1) koji smanjuje samo visinu
        """
        
        self.cnn = nn.Sequential(
            # Blok 1: 1 -> 32, H/2
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H: 48 -> 24

            # Blok 2: 32 -> 64, H/2
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),  # H: 24 -> 12

            # Blok 3: 64 -> 128, H/2
            ResidualBlock(64, 128),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: 12 -> 6, W ostaje

            # Blok 4: 128 -> 256
            ResidualBlock(128, 256),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: 6 -> 3, W ostaje

            # Blok 5: 256 -> 512
            ResidualBlock(256, 512),
            nn.MaxPool2d((3, 1), (3, 1)),  # H: 3 -> 1, W ostaje

            nn.Dropout2d(0.2),
        )

        # CNN izlaz: [B, 512, 1, W'] -> squash visine -> [B, 512, W']
        # Ulaz u LSTM: [W', B, 512]
        lstm_input_size = 512

        # ── BiLSTM deo ───────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=False,   # [T, B, features]
            bidirectional=True, #cita i sa leva na desno i sa desna na levo
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
        )

        # ── Izlazni sloj ─────────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 zbog bidirectional

    def forward(self, x):
        # x: [B, 1, H, W]
        features = self.cnn(x)               # [B, 512, 1, W']
        features = features.squeeze(2)       # [B, 512, W'] - izbacuje dimenziju visine
        features = features.permute(2, 0, 1) # [W', B, 512] = [T, B, input_size] - W postaje T(vreme)
                                             # svaka kolona slike postaje jedan vremenski korak

        lstm_out, _ = self.lstm(features)    # [T, B, hidden*2]
        logits = self.fc(lstm_out)           # [T, B, num_classes]

        return logits  # CTC loss očekuje [T, B, C]