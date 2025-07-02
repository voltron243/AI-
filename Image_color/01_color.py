# Creating a color image 
import numpy as np
import matplotlib.pyplot as plt

color_image = np.array(
    [
      [[0, 10, 255],[86, 65, 205]], 
      [[54, 75, 240],[230, 190, 200]]  
    ]   
)

plt.imshow(color_image)
plt.colorbar()
plt.show