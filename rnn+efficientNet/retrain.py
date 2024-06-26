import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

# Load the EfficientNet model
effnet_model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = effnet_model._fc.in_features  # Number of input features of the classifier layer
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

# Modify the LSTMModel class to include dropout after the LSTM layer
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Add dropout after the LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout
        out = self.fc(out)
        return out


class CombinedModel(nn.Module):
    def __init__(self, effnet_model, lstm_model):
        super(CombinedModel, self).__init__()
        self.effnet_model = effnet_model
        self.lstm_model = lstm_model
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

    def forward(self, x):
        with torch.no_grad():
            x = self.effnet_model.extract_features(x)  # Extract features using EfficientNet
        x = x.permute(0, 2, 3, 1)  # Permute dimensions to (batch_size, height, width, channels)
        x = x.view(x.size(0), x.size(1) * x.size(2), -1)  # Reshape to (batch_size, sequence_length, feature_size)
        x = self.dropout(x)  # Apply dropout
        x = self.lstm_model(x)  # Pass features through LSTM
        return x



# Initialize LSTM model
lstm_input_size = num_ftrs  # Same as the number of features extracted by EfficientNet
lstm_hidden_size = 128
lstm_num_layers = 1
lstm_num_classes = 2  # Assuming binary classification
lstm_model = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_num_classes).to(device)

# Initialize combined model
combined_model = CombinedModel(effnet_model, lstm_model).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

combined_model.load_state_dict(torch.load('combined_model_best.pth'))

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
            torch.save(model.state_dict(), 'model_best.pth')

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
train(combined_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Save the trained model
torch.save(effnet_model.state_dict(), 'model.pth')