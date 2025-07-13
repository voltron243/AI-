import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from CNN import CNN

"""
No need to modify
"""

def predict_image(model, image_path):

    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))

    image_array = np.array(image, dtype=np.float32) / 255.0

    if np.mean(image_array) > 0.5:
        image_array = 1.0 - image_array

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(image_path).convert('L'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_array, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')

    image_array = (image_array - 0.1307) / 0.3081

    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)

    # Prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        confidence = probabilities[0][predicted].item()

    plt.subplot(1, 3, 3)
    plt.bar(range(10), probabilities[0].numpy())
    plt.title(f'Predicted: {predicted.item()}\nConfidence: {confidence:.2f}')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()

    print(f"Predicted digit: {predicted.item()}")
    print(f"Confidence: {confidence:.2f}")
    
    return predicted.item()



if __name__ == "__main__":
    print("Running simple MNIST training...")

    model = CNN()
    model.load_state_dict(torch.load('/Users/andrechan/Documents/AI/mnist_model3.pth', map_location='cpu'))
    
    predicted_digit = predict_image(model, '/Users/andrechan/Documents/AI/Final_Project/photo.png')
