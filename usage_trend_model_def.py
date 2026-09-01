"""
Shared LSTM architecture definition for the usage-trend classifier.

Kept in its own module (rather than inline in train_usage_trend_model.py)
so both the training script and app.py can import the exact same class
when loading the saved weights - PyTorch's torch.load only restores
tensor values, not the class definition, so whatever loads the model
needs this class available.
"""

import torch.nn as nn


class UsageTrendLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len=5, input_size=1)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # last layer's hidden state -> class logits
        return out
