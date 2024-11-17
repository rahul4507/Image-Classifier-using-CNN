import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the CNN architecture
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

# Load the pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load("model/simple_cifar10_cnn.pth", map_location=torch.device("cpu")))
model.eval()  # Set the model to evaluation mode

# CIFAR-10 class names
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to match CIFAR-10 distribution
])

# Streamlit app UI
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="centered")
st.title("🚀 CIFAR-10 Image Classification")
st.write("""
Upload an image, and this application will predict its class using a pre-trained 
Convolutional Neural Network (CNN). Supported classes are based on the CIFAR-10 dataset.
""")

# Upload image section
st.sidebar.header("📂 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Processing your image...")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    # Display the prediction
    st.markdown(f"### 🎯 Prediction: **{label}**")
    st.balloons()  # Add some flair
else:
    st.sidebar.write("👈 Please upload an image to begin!")

# Additional information
st.sidebar.subheader("About")
st.sidebar.write("""
This app uses a simple CNN trained on CIFAR-10 to classify images. It is built with:
- **Streamlit** for the web interface
- **PyTorch** for deep learning
- **Pillow** for image processing
""")

st.sidebar.write("🔗 [Learn more about CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)")
