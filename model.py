import torch
import torch.nn as nn

# Define the model architecture
class PotholeModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PotholeModel, self).__init__()
        
        # Define the convolutional base model
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),  # Convolutional layer with 64 filters
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2),  # Max pooling layer with kernel size 2
            nn.Dropout(dropout_rate), # Dropout layer

            nn.Conv2d(64, 128, kernel_size=3),  # Convolutional layer with 128 filters
            nn.ReLU(),  
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate), # Dropout layer

            nn.Conv2d(128, 256, kernel_size=3),  # Convolutional layer with 256 filters
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive average pooling to reduce spatial dimensions
        )
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the convolutional layers
            nn.Linear(256*4*4, 1024),  # Linear layer with 1024 units
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout layer
        )
        
        # Define the final classification layer
        self.classifier = nn.Linear(1024, 2)  # Linear layer with 2 units for binary classification
    
    def forward(self, x):
        # Pass input through the convolutional base model
        h = self.base_model(x)
        
        # Pass output through fully connected layers
        h = self.fc_layers(h)
        
        # Pass output through the final classification layer
        y = self.classifier(h)
        
        # Return the output of the classification layer
        return y


import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import numpy as np

# # Dataset Class
# class PotholeDataset(Dataset):
#     def __init__(self, dataset_dir, csv_path, transform=None):
#         self.dataset_dir = dataset_dir
#         self.csv_path = csv_path
#         self.transform = transform
#         self.data = pd.read_csv(csv_path)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.dataset_dir, self.data.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")
#         label = int(self.data.iloc[idx, 1])  # Assuming labels are in the second column
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label

# # Model Definition
# class PotholeModel(nn.Module):
#     def __init__(self, dropout_rate=0.5):
#         super(PotholeModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(256 * 28 * 28, 512)  # Adjusted input size to match image size
#         self.fc2 = nn.Linear(512, 2)  # Output size should match the number of classes
#         self.dropout = nn.Dropout(dropout_rate)
#         self.pool = nn.MaxPool2d(2, 2)
    
#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))
#         x = self.pool(nn.ReLU()(self.conv3(x)))
        
#         # Flatten the output before feeding into the fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten the output to match the size (batch_size, features)
#         x = nn.ReLU()(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


# # Infinite DataLoader
# class InfiniteDataLoader:
#     def __init__(self, dataloader):
#         self.dataloader = dataloader
#         self.iter = iter(self.dataloader)
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         try:
#             return next(self.iter)
#         except StopIteration:
#             self.iter = iter(self.dataloader)
#             return next(self.iter)

# # Main script
# if __name__ == "__main__":
#     # Define dataset directory and CSV path
#     dataset_dir = r"C:\Users\John Paul\Desktop\ml_models\data"  
#     csv_path = os.path.join(dataset_dir, 'pothole_labels.csv')  

#     # Data augmentation and transformations
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize all images to 224x224
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Initialize dataset and data loader
#     dataset = PotholeDataset(dataset_dir, csv_path, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     # Create an infinite data loader
#     infinite_loader = InfiniteDataLoader(dataloader)

#     # Initialize model, criterion, and optimizer
#     model = PotholeModel(dropout_rate=0.5)  # Added dropout
#     criterion_classification = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)  # Adjusted lr and added weight decay

#     # Training loop with early stopping
#     num_epochs = 10
#     best_f1 = 0.0
#     patience = 3
#     counter = 0

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for images, labels in infinite_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion_classification(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             # Calculate accuracy
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         epoch_loss = running_loss / len(dataloader)
#         epoch_accuracy = correct / total
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
#         # Save the model if it performs better
#         if epoch_accuracy > best_f1:
#             best_f1 = epoch_accuracy
#             counter = 0
#             # Save model and optimizer states
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
#             }, 'best_model.pth')  # Save the best model
#             print("Model saved as 'best_model.pth'")
#         else:
#             counter += 1
        
#         # Early stopping
#         if counter >= patience:
#             print("Early stopping triggered")
#             break

#     print("Training finished")

# # Loading the saved model
# # After training or when you need to use the model later, load the saved checkpoint.
# checkpoint = torch.load('best_model.pth')

# # Initialize the model and optimizer
# model = PotholeModel(dropout_rate=0.5)  # Use the same model architecture
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)

# # Load model state dict and optimizer state dict
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # Load the epoch and loss if resuming training
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# # Set the model to evaluation mode for inference
# model.eval()
# print(f"Model loaded and ready for inference from epoch {epoch}.")
