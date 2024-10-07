import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from model.u_net import Unet
from model.forward import ForwardDiffusion

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(10,10)) 
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0], cmap='gray')
        print(img[0])
    plt.show()

#data = torchvision.datasets.FashionMNIST('./data', download=True, train=True)
#show_images(data)


def load_transformed_dataset(train=True):
    data_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    if train:
        data = torchvision.datasets.CIFAR10(root="./data", download=True, 
                                            transform=data_transform, train=True)

    else:
        data = torchvision.datasets.CIFAR10(root="./data", download=True, 
                                            transform=data_transform, train=False)


    return data

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=2, shuffle=True, drop_last=True)

T=100
forward = ForwardDiffusion(timesteps=T, scedule_type="linear")




# Simulate forward diffusion
image = next(iter(dataloader))[0]


plt.figure(figsize=(12, 5))
plt.axis('off')
num_images = 7
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx, 1]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    plt.axis('off')
    img, noise = forward.q_sample(image, t)

    show_tensor_image(img)

plt.show()
