import torch
from torchvision import transforms  # Import transforms
from util.dfnn_model_lstm import CNNLSTM
from util.train_model_lstm import train_with_validation, validate
from util.load_data_lstm import load_dataset

# Hyperparameters
sequence_length = 5
batch_size = 8
num_epochs = 10
learning_rate = 0.001
data_dir = "./LSTMDataset"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CNNLSTM(num_classes=2, sequence_length=sequence_length).to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor()
])

# Load datasets
train_loader = load_dataset(data_dir=f"{data_dir}/Train/Real", sequence_length=sequence_length, batch_size=batch_size, transform=transform)
val_loader = load_dataset(data_dir=f"{data_dir}/Validation/Real", sequence_length=sequence_length, batch_size=batch_size, transform=transform)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Starting training...")
train_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

# Test the model
print("Testing the model...")
test_loader = load_dataset(data_dir=f"{data_dir}/Test/Real", sequence_length=sequence_length, batch_size=batch_size, transform=transform)
avg_loss, accuracy = validate(model, test_loader, criterion, device)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
