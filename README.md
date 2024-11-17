# README: Image Classification with CNN using Streamlit

## Overview
This project demonstrates an image classification application using a pre-trained Convolutional Neural Network (CNN). The application classifies images into one of the 10 CIFAR-10 categories using a web-based user interface built with Streamlit.

## Features
- **Pre-Trained CNN**: Uses a simple CNN architecture pre-trained on the CIFAR-10 dataset.
- **Streamlit UI**: Provides an intuitive interface for uploading and classifying images.
- **Real-Time Prediction**: Displays the predicted class for uploaded images.

## Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Web application framework.
- **PyTorch**: Deep learning framework for model creation and inference.
- **Pillow (PIL)**: Library for image processing.

## Prerequisites
- Python 3.8 or higher
- Required Python packages (listed below)

## Installation and Setup
1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   streamlit
   torch
   torchvision
   Pillow
   ```

3. **Prepare the Model**  
   Place the pre-trained model file (`simple_cifar10_cnn.pth`) in the `model/` directory. You can train this model using the CIFAR-10 dataset or download a pre-trained version.

4. **Run the Application**  
   Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. **Access the Web App**  
   Open your browser and navigate to the URL provided by Streamlit (e.g., `http://localhost:8501`).

## Usage
1. Upload an image in **JPEG**, **PNG**, or **JPG** format using the file uploader.
2. The app will display the uploaded image.
3. The model will classify the image into one of the following CIFAR-10 categories:
   - Airplane
   - Car
   - Bird
   - Cat
   - Deer
   - Dog
   - Frog
   - Horse
   - Ship
   - Truck
4. The predicted class will be displayed below the image.

## Project Structure
```
project/
├── app.py               # Main application script
├── model/
│   └── simple_cifar10_cnn.pth  # Pre-trained model file
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Customization
- **Model Architecture**: Modify the `SimpleCNN` class in `app.py` to experiment with different CNN architectures.
- **Dataset**: Retrain the model on custom datasets if required.
- **UI Enhancements**: Customize the Streamlit UI as needed.

## Limitations
- The model is trained on the CIFAR-10 dataset, which includes small (32x32 pixel) images. Performance may degrade on images that are very different from this dataset.

