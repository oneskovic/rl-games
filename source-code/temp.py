import numpy as np
import torch.nn.functional as F
import torch
a = torch.tensor([[[
[0,2,2,7],
[3,2,5,8],
[6,7,8,4],
[6,2,2,1]]]])
b = torch.tensor([[[
[0,1],
[2,3]
]]])
output = F.conv2d(a, b, stride=2, padding=0)
output = output.numpy()[0][0]
print(output)