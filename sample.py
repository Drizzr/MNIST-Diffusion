from model.forward import ForwardDiffusion
from model.u_net import Unet
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import matplotlib.animation as animation



parser = argparse.ArgumentParser(description="Sample from the model")

parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sampling, must be a square number")
parser.add_argument("--timesteps", type=int, default=200, help="Number of timesteps")
parser.add_argument("--guidance", type=float, default=7, help="guidance value")
parser.add_argument("--c", type=int, default=1, help="Class to sample from")
parser.add_argument("--model_path", type=str, default="checkpoints/checkpoint_epoch_41_0.0%_estimated_loss_0.035/model.pth", help="Path to the model")
parser.add_argument("--animate", action='store_true',
                        default=True, help="Animate the diffusion process, for a single sample")
parser.add_argument("--img_size", type=int, default=28, help="Image size")
parser.add_argument("--channels", type=int, default=1, help="Number of channels")


args = parser.parse_args()

model = Unet(
    dim=args.img_size,
    channels=args.channels,
    )

BATCH_SIZE = args.batch_size
time_steps = args.timesteps
guidance = args.guidance
class_ = args.c
animate = args.animate

print("Sampling from the model...")


model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()


forward = ForwardDiffusion(timesteps=time_steps, start=0.0001, end=0.02)

samples = forward.sample(model, image_size=28, batch_size=BATCH_SIZE, channels=1, class_=torch.tensor([class_]), guidance=guidance) 


# create a grid of 8x8 images
fig, ax = plt.subplots(int(np.sqrt(BATCH_SIZE)), int(np.sqrt(BATCH_SIZE)), figsize=(10, 10))
for i in range(int(np.sqrt(BATCH_SIZE))):
    for j in range(int(np.sqrt(BATCH_SIZE))):
        if args.channels == 1:
            ax[i, j].imshow(samples[-1][i*int(np.sqrt(BATCH_SIZE))+j].reshape(args.img_size, args.img_size, args.channels), cmap="gray")
        else:
            ax[i, j].imshow(samples[-1][i*int(np.sqrt(BATCH_SIZE))+j].reshape(args.img_size, args.img_size, args.channels))


fig.savefig("samples.png")


if animate:

    random_index = random.randint(0, BATCH_SIZE-1)

    fig = plt.figure()
    ims = []
    for i in range(0, 200):
        if args.channels == 1:
            im = plt.imshow(samples[i][random_index].reshape(args.img_size, args.img_size, args.channels), cmap="gray", animated=True)

        else:
            im = plt.imshow(samples[i][random_index].reshape(args.img_size, args.img_size, args.channels), animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')

plt.show()