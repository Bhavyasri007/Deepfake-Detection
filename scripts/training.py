import os
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # If CUDA is available, set the device to GPU
    device = torch.device("cuda")
    print("CUDA (GPU) is available. Training on GPU...")
else:
    # If CUDA is not available, set the device to CPU
    device = torch.device("cpu")
    print("CUDA (GPU) is not available. Training on CPU...")

# Define data directories
data_dir = "D:\\deepfake project\\data"
train_dir = os.path.join(data_dir, "frames", "classified sets", "training_set")
val_dir = os.path.join(data_dir, "frames", "classified sets", "validation_set")
test_dir = os.path.join(data_dir, "frames", "classified sets", "testing_set")

# Define transforms (optional)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224 (example)
    transforms.ToTensor(),           # Convert frames to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize frames
])

# Define datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Define data loaders
batch_size = 32  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
print(len(train_loader.dataset))  # Print number of samples in training set
print(len(val_loader.dataset))    # Print number of samples in validation set
print(len(test_loader.dataset))   # Print number of samples in testing set
'''
'''
# Get a batch of samples from the data loader
batch_images, batch_labels = next(iter(train_loader))

# Display the first few sample frames
for i in range(5):
    plt.imshow(batch_images[i].permute(1, 2, 0))  # Convert tensor to image format (C, H, W) -> (H, W, C)
    plt.title(f"Label: {batch_labels[i]}")
    plt.show()
'''
# Define your EfficientNet model
effnet_model = EfficientNet.from_pretrained('efficientnet-b0')

# Access the last layer before the classifier to get the number of input features
num_ftrs = effnet_model._fc.in_features  # Access the number of input features of the classifier layer
print(num_ftrs)  # Print the number of input features

class CustomClassifier(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.dropout1 = nn.Dropout(0.5)  # Add dropout after the first fully connected layer
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # Add dropout after the second fully connected layer

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x



# Replace the existing classifier with the custom one
effnet_model._fc = CustomClassifier(num_ftrs, 2)


# Define loss function (Binary Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Define optimizer (e.g., Adam)
optimizer = optim.Adam(effnet_model.parameters(), lr=0.001)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Print statistics for the current epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_accuracy:.4f}")

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'efficientnet_model_best.pth')

def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update statistics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct_predictions / total_samples
    return avg_val_loss, val_accuracy

# Assuming 'device' is properly defined (e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Train the model
train(effnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Save the trained model
torch.save(effnet_model.state_dict(), 'efficientnet_model.pth')



