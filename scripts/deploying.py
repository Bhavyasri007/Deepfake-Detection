import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

device = torch.device("cpu")
# Load the trained model
effnet_model = EfficientNet.from_pretrained('efficientnet-b0')
effnet_model.eval()  # Set the model to evaluation mode
num_ftrs = effnet_model._fc.in_features  # Access the number of input features of the classifier layer

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
effnet_model.load_state_dict(torch.load("efficientnet_model.pth", map_location=device))
effnet_model.to(device)  # Move the model to the device

# Define transforms for preprocessing images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define function to preprocess image
def preprocess_image(image):
    image = preprocess(image).unsqueeze(0)
    return image

# Define an endpoint to accept image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        image = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            image = image.to(device)
            outputs = effnet_model(image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
