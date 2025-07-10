import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from CNN import CNN
from plot import plot_training_history
from dataloader import MNISTDataLoader

# Training model
# Training process has nothing to incapsilate (no class is needed)
# Define loss function
def train_model(model, train_loader):    
    loss_func = nn.CrossEntropyLoss() # Calculate the loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # List
    train_loss = [] # loss is matrces 
    train_acc = [] # accuracy 
    
    # Put the model into the training model
    for epoch in range(30): # Example: 10 epochs
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        # Loop
        # Most important steps 
        # Always need to clear the gradient.
      
        for image, label in pbar:
            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step() # Update the whole network
            
            running_loss += loss.item() # 
            _, prediction = torch.max(output.data, 1)
            total += label.size(0)
            correct += (prediction == label).sum().item()
            
            pbar.set_postfix(loss=running_loss / total, accuracy=correct / total)
        
        # Append is a= [1,2,3] -> a.append (1). -> a = [1,2,3,1]
        # Its about how much epoches you have 
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total 
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # See what actually is in it 
        # programming starts from 0 then 1,2 and so on
        print(f"Epoch [{epoch+1}/30], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% ")
        
    return train_loss, train_acc

# Observe how many cases have been correct, including loss
def test_model(model, test_loader):
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    loss_func = nn.CrossEntropyLoss() # Calculate the loss
    
    
    with torch.no_grad(): # No need to calculate the gradient
        for image, label in test_loader:
            
            output = model(image)
            loss = loss_func(output, label)
            
            test_loss += loss.item()
            _, prediction = torch.max(output.data, 1) # Prediction
            
            total += output.size(0)
            correct += (prediction == label).sum().item()
            
    test_loss = test_loss / len(test_loader) # Test loss
    test_acc = 100. * correct / total # 
    
    print(f"Test loss: {test_loss:.f} Accuracy: {test_acc:.4f}%")
            
# Entrance of the whole project
if __name__ == "__main__":
    data_loader = MNISTDataLoader()
    
    train_loader, test_loader = data_loader.get_data_loaders() # Instance 
    
    print(train_loader, test_loader) # Memory location on the ram 
    
    # Create an instance from the model 
    model = CNN()
    
    print(model)
    
    # Train 
    train_loss, train_acc = train_model(model, train_loader)
    
    test_model(model, test_loader)
    
    plot_training_history(train_loss, train_acc)
    
    # save the model as a file (*.pth)
    torch.save(model.state_dict(), 'mnist_model.pth')