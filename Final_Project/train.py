import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Training process has nothing to incapsilate (no class is needed)
# Define loss function
def train_model(model, train_loader):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # List
    train_loss = [] # loss is matrces 
    train_acc = []
    
    # Put the model into the training model
    for epoch in range(30): # Example: 10 epochs
        model.train()
        
        pbar = tqdm(train_loader, desc="Training")
        
        # Loop
        # Most important steps 
        # Always need to clear the gradient. They 
        for image, label in pbar:
            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
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
        print(f"Epoch {[epoch+1]/30}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% ")
        
    return train_loss, train_acc
        
def test_model():
    pass



if __name__ == "__main__":
    pass