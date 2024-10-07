from model.forward import ForwardDiffusion
from model.u_net import Unet
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import matplotlib.animation as animation
import os
import json



parser = argparse.ArgumentParser(description="Sample from the model")

parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sampling, must be a square number")
parser.add_argument("--timesteps", type=int, default=200, help="Number of timesteps")
parser.add_argument("--guidance", type=float, default=7, help="guidance value")
parser.add_argument("--c", type=int, default=0, help="Class to sample from")
parser.add_argument("--model_path", type=str, default="checkpoints_CIFAR10/checkpoint_epoch_3_0.0%_estimated_loss_0.037", help="Path to the model")
parser.add_argument("--animate", action='store_true',
                        default=True, help="Animate the diffusion process, for a single sample")

args = parser.parse_args()

with open(os.path.join(args.model_path, "params.json"), "r") as f:
    params = json.load(f)

try:
    
    img_size = params["img_size"]
    channels = params["channels"]
    n_classes = params["n_classes"]
    dim_mults = params["dim_mults"]

except Exception as e:
    print(e)
    print("params.json incomplete, reverting to default values")
    img_size = 28
    channels = 1
    n_classes = 10
    dim_mults = (1, 2, 4,)


model = Unet(
    dim=img_size,
    channels=channels,
    dim_mults=dim_mults,
    n_classes=n_classes
    )

BATCH_SIZE = args.batch_size
time_steps = args.timesteps
guidance = args.guidance
class_ = args.c
animate = args.animate

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Sampling from the model on {DEVICE}...")


model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth"), map_location=DEVICE))
model.eval()

model.to(DEVICE)


forward = ForwardDiffusion(timesteps=time_steps, start=0.0001, end=0.02)

samples = forward.sample(model, image_size=img_size, batch_size=BATCH_SIZE, channels=channels, class_=torch.tensor([class_]).to(DEVICE), guidance=guidance) 


# create a grid of 8x8 images
fig, ax = plt.subplots(int(np.sqrt(BATCH_SIZE)), int(np.sqrt(BATCH_SIZE)), figsize=(10, 10))
for i in range(int(np.sqrt(BATCH_SIZE))):
    for j in range(int(np.sqrt(BATCH_SIZE))):
        samples[-1][i*int(np.sqrt(BATCH_SIZE))+j] = np.clip((samples[-1][i*int(np.sqrt(BATCH_SIZE))+j] + 1) / 2, 0, 1)
        if channels == 1:
            ax[i, j].imshow(samples[-1][i*int(np.sqrt(BATCH_SIZE))+j].reshape(img_size, img_size, channels), cmap="gray")
        else:
            ax[i, j].imshow(samples[-1][i*int(np.sqrt(BATCH_SIZE))+j].reshape(img_size, img_size, channels), cmap="viridis")


fig.savefig("samples.png")


if animate:

    random_index = random.randint(0, BATCH_SIZE-1)

    fig = plt.figure()
    ims = []
    for i in range(0, 200):
        samples[i][random_index] = np.clip((samples[i][random_index] + 1) / 2, 0, 1)
        if channels == 1:
            im = plt.imshow(samples[i][random_index].reshape(img_size, img_size, channels), cmap="gray", animated=True)
        else:
            im = plt.imshow(samples[i][random_index].reshape(img_size, img_size, channels), animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')

plt.show()