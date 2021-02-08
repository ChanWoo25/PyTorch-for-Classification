
# %% IMPORT
import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

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

#%%
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

trainset = datasets.STL10(root='./data/', split='train', transform=data_transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

testset = datasets.STL10(root='./data/', split='test', transform=data_transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

classes = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")

#%%

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#%%

net = torchvision.models.alexnet(pretrained=False, num_classes=10)
criterion = torch.nn.L1Loss(reduction='sum')
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


net.apply(init_weights)



for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        print(f"input's shape: {inputs.shape}, labels' shape: {labels.shape}")
        opt.zero_grad()

        outputs = net(inputs)
        outputs = outputs.argmax(dim=1)
        print(f"outputs's shape: {outputs.shape}, labels' shape: {labels.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')















#%%
# torch.utils.data.Dataset 추상 클래스를 상속받아 커스텀 데이터셋 생성
# __init__, __len__, __getitem__ 3가지 메소드를 오버라이딩함

# class Custom(torch.utils.data.Dataset):
#     def __init__(self):
#         """데이터셋의 전처리를 담당"""
#         self.x = 10
#
#     def __len__(self):
#         """데이터셋의 길이, 총 샘플의 수를 적어줌 == len(dataset)"""
#
#     def __getitem__(self, idx):
#         """데이터셋에서 특정 1개의 샘플을 가져오는 함수 == dataset[i]"""


















