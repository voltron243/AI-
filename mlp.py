import torch
import torch.nn as nn
import torch.nn.functional as F

# Making a class MLP is inherited from nn.module 
# Based on this module
# Creating a class 
# Inherited the vaues from the parent class
# Always write the super line (function)

# Define a simple Multi - Layer Perceptron (MLP) class
# Parent class in Pytorch
# (PREDICTION FUNCTION)

# 1. Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        # Define the layer of the MLP
        self.fc1 = nn.Linear(input_size, hidden_size) # First fully connected layer 
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer 
    
    # Must activate it by 1
    # How to use these layers
    def forward(self, x):
        y1 = self.fc1(x) # Pass input through the first layer 
        activated_y1 = F.relu(y1) # Apply ReLU activation function
        
        y2 = self.fc2(activated_y1) # Pass through the second layer
        
        return y2 # Return the output of the second layer
    
# 2. Create an instance of the MLP class/create the network 
input_size = 784 
hidden_size = 128 # 2^7. 0, 1  true & false, 2, 2^n
output_size = 2

# Making it an instance 
model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

print(model)

# 3. Define a sample input tensor
sample_input = torch.randn(1, input_size) # Batched size of 1
output = model(sample_input) # Forward pass through the model 

print("Sample output:", output)


        