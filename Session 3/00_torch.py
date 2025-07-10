import torch

tensor0 = torch.tensor([1, 2, 3])

print(tensor0)

# 0 Matrix
# creating three rows and four collumns
zeros_tensor = torch.zeros(3, 4)
print(zeros_tensor)

# 1 Matrix
ones_tensor = torch.ones(3, 4)
print(ones_tensor)

# random
# Matrix that follows random numbers 
# Between 0 to 1
# int in sizes
random_tensor = torch.randint(1, 10, [3, 4])
print(random_tensor)

# 1 + 4, 2 + 5, 3 + 6
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a * b
print(c)
# dot product 