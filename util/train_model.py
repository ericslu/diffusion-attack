import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    print("Starting training...")  # Inform the user that training is starting
    
    # Create a progress bar for the entire training process
    for epoch in range(num_epochs):
        total_loss = 0
        # Create a progress bar for each epoch's dataloader iteration
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update the progress bar with the latest loss value
                pbar.set_postfix(loss=total_loss / (pbar.n + 1), refresh=True)
        
        # Print average loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
        
    print("Training complete!")  # Inform the user that training has finished
