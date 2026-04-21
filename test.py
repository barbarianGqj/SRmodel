import torch

"""
tensor1 = torch.rand(1,2,3,4,5)
print(tensor1)
print(tensor1.size())
tensor2 = tensor1.squeeze(0)
print(tensor2)
print(tensor2.size())
tensor3 = tensor1.unsqueeze(3)
print(tensor3)
print(tensor3.size())
tensor4 = tensor1.unsqueeze(5)
print(tensor4)
print(tensor4.size())
"""

"""
h = 3
w = 5
grid_y, grid_x = torch.meshgrid(
    torch.arange(0, h),
    torch.arange(0, w))
grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
print(grid_y)
print(grid_y.size())
print(grid_x)
print(grid_x.size())
print(grid)
print(grid.size())
"""

idx = 1
tensor1 = torch.randn(3,5,6) # [3, 5, 6]
tensor2 = tensor1[..., idx:idx + 2] # [3, 5, 2] 只取dim = 2的idx+1, idx+2列
print(tensor1)
print(tensor1.size())
print(tensor2)
print(tensor2.size())