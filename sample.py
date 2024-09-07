from model.forward import ForwardDiffusion
from model.u_net import Unet
import torch
import matplotlib.pyplot as plt


forward = ForwardDiffusion(timesteps=200, start=0.0001, end=0.02)

model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4,)
    )

model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_2_105.68%_estimated_loss_0.062/model.pth"))

samples = forward.sample(model, image_size=28, batch_size=1, channels=1, class_=torch.tensor([9]), w=3)


plt.imshow(samples[-1][0].reshape(28, 28, 1), cmap="gray")
plt.show()