import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Definition of class
class MNISTDataLoader:
    def __init__(self):
        super().__init__()
        
        # transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transforms
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transforms
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    
    def show_sample(self):
        image, label = self.train_dataset[4540]
        image = image.squeeze().numpy()  # Convert to numpy array
        image = image * 0.3081 + 0.1307  # Denormalize
        
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    
    data_loader = MNISTDataLoader()
    
    print(len(data_loader.train_dataset))
    
    train_loader, test_loader = data_loader.get_data_loaders()
    
    data_loader.show_sample()


