import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectorCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.cnn(x)