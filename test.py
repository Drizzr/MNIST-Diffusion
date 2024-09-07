import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from model.u_net import SimpleUnet
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


def load_transformed_dataset():
    data_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.FashionMNIST(root="./data", download=True, 
                                        transform=data_transform, train=True)

    test = torchvision.datasets.FashionMNIST(root="./data", download=True, 
                                        transform=data_transform, train=False)
    return torch.utils.data.ConcatDataset([train, test])

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
    plt.imshow(reverse_transforms(image), cmap='gray')

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=2, shuffle=True, drop_last=True)

T=100
forward = ForwardDiffusion(timesteps=T, scedule_type="linear")




# Simulate forward diffusion
image = next(iter(dataloader))[0]


plt.figure(figsize=(12, 5))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx, 1]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward.forward_diffusion_sample(image, t)

    show_tensor_image(img)

plt.show()

model = SimpleUnet()

model.load_state_dict(torch.load("checkpoints/chechpoint_epoch_1_56.8%_estimated_loss_0.799/model.pth"))


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = forward.get_index_from_list(forward.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = forward.get_index_from_list(
        forward.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = forward.get_index_from_list(forward.sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, torch.tensor([0])) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = forward.get_index_from_list(forward.posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = 28
    img = torch.randn((1, 1, img_size, img_size))
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()        

sample_plot_image()