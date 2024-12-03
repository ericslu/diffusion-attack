import torch

def train_with_validation(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs in train_loader:
            inputs = inputs.to(device)
            labels = torch.randint(0, 2, (inputs.size(0),)).to(device)  # Dummy labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

        if validation_loader:
            validate(model, validation_loader, criterion, device)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            labels = torch.randint(0, 2, (inputs.size(0),)).to(device)  # Dummy labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
