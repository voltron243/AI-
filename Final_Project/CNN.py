import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of class
# Convolutional as Conv
class CNN(nn.Module):
    def __init__(self, cls_num=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
        # Define the fully connected layer (MLP)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # 3134
        self.fc2 = nn.Linear(128, cls_num) # Assuming 10 output
    
    # Always use reLU in the fucntional language 
    def forward(self, x):
        # first conv block (layer)
        y1 = self.conv1(x)
        y1 = F.relu(y1)
        y1 = self.pool1(y1)
        
        # second conv block (layer)
        y2 = self.conv2(y1)
        y2 = F.relu(y1)
        y2 = self.pool2(y2) 
        
        f = y2.view(y2.size(0), -1)
        
        z = F.relu(self.fc1(f))
        z = self.dropout(z)
        z = self.fc2(z)
    
        return z


# Entrance
if __name__ == "__main__":
    model = CNN(cls_num=10) # Create an instance of the CNN class
    print(model) # Print the model architecture
    
    import_tensor = torch.randn(1, 1, 28, 28) # Batch size of 1, 1 channel, 28x28 image
    