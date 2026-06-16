"""
CRNN model za OCR sa Dropout regularizacijom.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=48, hidden_size=256, num_lstm_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        cnn_h = img_height // 16
        self.fc = nn.Linear(256 * cnn_h, hidden_size)
        self.dropout = nn.Dropout(0.4)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=0.4 if num_lstm_layers > 1 else 0,
            batch_first=False
        )

        self.fc_out = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.shape
        conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out = torch.relu(self.fc(conv))
        out = self.dropout(out)
        out = out.permute(1, 0, 2)
        out, _ = self.lstm(out)
        return self.fc_out(out)