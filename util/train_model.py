import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar

# def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
#     model.train()
    
#     print("Starting training...")  # Inform the user that training is starting
    
#     # Create a progress bar for the entire training process
#     for epoch in range(num_epochs):
#         total_loss = 0
#         # Create a progress bar for each epoch's dataloader iteration
#         with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as pbar:
#             for images, labels in pbar:
#                 images, labels = images.to(device), labels.to(device)
                
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
                
#                 # Update the progress bar with the latest loss value
#                 pbar.set_postfix(loss=total_loss / (pbar.n + 1), refresh=True)
        
#         # Print average loss for the epoch
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
        
#     print("Training complete!")  # Inform the user that training has finished

def train_with_validation(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in train mode
        total_train_loss = 0
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training:")
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validate the model after each epoch
        avg_val_loss, val_accuracy = validate(model, validation_loader, criterion, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode for validation
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test(model, dataloader, criterion, device):
    return validate(model, dataloader, criterion, device)