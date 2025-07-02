# Creating a grey image 
import numpy as np
import matplotlib.pyplot as plt

# value range : 0-255
grey_image = np.array(
    [
        [125, 20, 199, 0, 26],
        [0, 20, 199, 0, 26],
        [0, 20, 199, 0, 26],
        [0, 22, 160, 0, 27],
        [0, 20, 199, 0, 26]
    ]
)
# Argument 
plt.imshow(grey_image, cmap='gray' )
plt.show()