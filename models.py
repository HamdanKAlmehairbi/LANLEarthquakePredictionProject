import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_features, num_classes=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(0.01)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5, 50)  # Seq length 10 -> pooled to 5
        self.relu3 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # Input: (batch, seq_len, features) -> permute for Conv1d
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_units=64, num_layers=2, num_classes=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_features, hidden_size=hidden_units,
            num_layers=num_layers, batch_first=True,
            dropout=0.4, bidirectional=True  # Increased from 0.2 to 0.4 for better regularization
        )
        self.fc1 = nn.Linear(hidden_units * 2, 32)  # Input size is doubled
        self.relu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        last_step_output = output[:, -1, :] # Use output from the last time step
        x = self.relu(self.fc1(last_step_output))
        x = self.fc2(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, input_features, num_classes=1):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=80, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(80 * 2, 40) # Input size doubled
        self.relu2 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(40, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        last_step_output = output[:, -1, :]
        x = self.relu2(self.fc1(last_step_output))
        x = self.fc2(x)
        return x


class HybridAttention(nn.Module):
    def __init__(self, input_features, attn_dim=64, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=80, num_layers=1, batch_first=True, bidirectional=True)
        # Additive attention layers
        self.attn_w = nn.Linear(80 * 2, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1)
        self.fc1 = nn.Linear(80 * 2, 40)
        self.relu2 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(40, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)           # (B, T, H*2)
        # Apply attention mechanism
        e = torch.tanh(self.attn_w(out)) # (B, T, A)
        scores = self.attn_v(e).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)  # (B, H*2)
        
        x = self.relu2(self.fc1(context))
        return self.fc2(x)


