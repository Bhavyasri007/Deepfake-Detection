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


def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate test loss and accuracy
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct_predictions / total_samples
    return avg_test_loss, test_accuracy

# Load the trained model
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

effnet_model._fc = CustomClassifier(num_ftrs, 2)  # Replace the classifier with the custom one
effnet_model.load_state_dict(torch.load("efficientnet_model.pth"))
effnet_model.to(device)  # Move the model to the device
effnet_model.eval() 

# Define loss function (Binary Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test dataset
test_loss, test_accuracy = evaluate_model(effnet_model, test_loader, criterion, device)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
