import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
from model import PotholeModel

# Load the model
model = PotholeModel()
model.load_state_dict(torch.load('cnn_enhanced.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        pred_class = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        one_hot[0][pred_class] = 1
        output.backward(gradient=one_hot)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)  # Normalize heatmap to range [0, 1]
        
        return heatmap

# Streamlit app
st.title('Pothole Detection App')

# File uploader
uploaded_file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load the image
    img = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)
    
    # Perform prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    # Compute confidence
    confidence = torch.softmax(output, dim=1)[0, predicted[0]].item()
    
    # Grad-CAM
    target_layer = model.base_model[2]  # Targeting the third convolutional layer
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(img_tensor)
    
    # Convert heatmap to a format that can be displayed in Streamlit
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    # Superimpose heatmap on the image
    img = np.array(img)
    superimposed_img = cv2.addWeighted(img, 1, heatmap, 0.4, 0)

    # Display results
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"Prediction: {'Pothole detected' if predicted[0] else 'No pothole detected'}")
    st.write(f"Confidence: {confidence}")
    
    # Display the original image and Grad-CAM
    st.image(img, caption="Original Image", use_column_width=True)
    st.image(superimposed_img, caption="Grad-CAM Output", use_column_width=True)
