import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the CNN architecture (must match the model that created the .pth file)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("model/simple_cifar10_cnn.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# Define the CIFAR-10 class names
classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Streamlit UI code
st.title("Image Classification with CNN")
st.write("Upload an image, and the model will classify it based on CIFAR-10 classes.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and display it
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    # Display the result
    st.write(f"Prediction: **{label}**")
