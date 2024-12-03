import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super(CNNLSTM, self).__init__()
        self.sequence_length = sequence_length
        
        # Load a pretrained ResNet
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()  # Replace the final fully connected layer with an identity
        self.feature_size = 512  # ResNet18's feature output size
        
        # Define LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_size, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        
        # Define a fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # Input shape: (B, T, C, H, W)
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten sequence for CNN
        features = self.cnn(x)  # Output shape: (B*T, feature_size)
        features = features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(features)  # Output shape: (B, T, hidden_size)
        output = self.fc(lstm_out[:, -1, :])  # Take last LSTM output
        return output
