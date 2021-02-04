
# %% IMPORT
import torch
import numpy as np

# %% TENSOR
data = [[1, 2], [3, 4]]
print(data)
x_data = torch.tensor(data)
print(x_data)
np_arr = np.array(data)
print(np_arr)

x_np = torch.from_numpy(np_arr)
print(x_np)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=float)
print(x_rand)

shape = (2, 3, )
randT = torch.rand(shape)
onesT = torch.ones(shape)
zeroT = torch.zeros(shape)

print(randT, onesT, zeroT, sep='\n')
print(randT.shape, randT.dtype, randT.device)

# %% TENSOR OPERATION

t0 = torch.Tensor([[1, 2], [3, 4]])
if torch.cuda.is_available():
    t0 = t0.to('cuda')

print(t0, f"\n{t0.shape}, {t0.dtype}, {t0.device}")
t1 = torch.cat([t0, t0], dim=1)
print(t1)
t1 = torch.cat([t1, t1], dim=0)
print(t1)

# element-wise product
print(t0.mul(t0))
print(t0 * t0)

# Matrix multiplication
print(t0.matmul(t0.T))
print(t0 @ t0.T)

# "_ suffix" operation denotes in-place.
t0.add_(1)
print(t0)

if not t0.is_quantized:
    print("True")

t2 = torch.tensor([1, 2], dtype=torch.int)
print(t2.is_quantized)


