
# Diffusion Model with Classifier-Free Guidance

This repository contains an implementation of the **Denoising Diffusion Probabilistic Model (DDPM)** algorithm with **classifier-free guidance**. It is designed for training and sampling on the **Fashion-MNIST** and **CIFAR-10** datasets using a flexible **UNet architecture**.

## Model Overview

- The **DDPM** algorithm follows the methodology described in the original DDPM paper.
- The **UNet architecture** is largely adapted from Hugging Face’s diffusion notebook: [annotated diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb).
- Classifier-free guidance is implemented based on the approach described in [this paper](https://arxiv.org/pdf/2207.12598).

## How to Train

To train the model on the **Fashion-MNIST** dataset, run the following command:
```bash
python3 train.py
```

### Training Options:
- `--num_epochs`: Number of training epochs (default: 5).
- `--batch_size`: Size of the mini-batch for each training iteration (default: 8).
- `--dropout`: Dropout rate to apply during training (default: 0.5).
- `--print_freq`: Frequency of print updates during training (default: 100 iterations).
- `--p_uncond`: Probability of unconditional sampling (default: 0.2).
- `--timesteps`: Number of timesteps for diffusion training (default: 200).
- `--from_check_point`: Add this flag if you wish to continue training from a saved checkpoint.
- `--clip`: Gradient clipping value (default: 10.0).
- `--max_lr`: Maximum learning rate (default: 4e-4).
- `--load_dir`: Directory of a saved model checkpoint.
- `--save_dir`: Directory where the model will be saved (default: "checkpoints/").
- `--img_size`: Size of the square image (default: 28).
- `--channels`: Number of channels in the image (default: 1).
- `--n_classes`: Number of classes in the dataset (default: 10).
- `--dim_mults`: Dimension multipliers for the UNet (default: "(1,2,4)"). Ensure the image size is divisible by the last multiplier.

### Training Progress:
During training, the model's loss decreases steadily as shown in the following plot, which reflects the improvement in performance over time:

![image](https://github.com/user-attachments/assets/65315b33-251f-4235-8835-8dc42948b9a6)

## How to Sample

To sample from a trained model, run the following command:
```bash
python3 sample.py
```

### Sampling Options:
- `--c [0-9]`: Specify the class (0-9) for class-conditional sampling.
- `--guidance float`: Guidance strength. The default value (`w=8`) works well, though this may need adjustment for different classes.
- `--model_path`: Load a model checkpoint. Based on previous results, checkpoint from epoch 40 works best.
- `--batch_size`: Specify the number of images to sample. Must be a square number for visualization (default: 4).
- `--animate`: Add this flag to visualize the reverse diffusion process for one random image in the batch.
- `--timesteps`: Number of timesteps to sample over (default: 200).

## Customizing the Dataset

This repository can be used to train an arbitrary classifier-free guided diffusion model. To train on a different dataset, modify the `load_transformed_dataset` function in `train.py`:

```python
def load_transformed_dataset(train=True):
    data_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    if train:
        data = torchvision.datasets.FashionMNIST(root="./data", download=True, 
                                                 transform=data_transform, train=True)
    else:
        data = torchvision.datasets.FashionMNIST(root="./data", download=True, 
                                                 transform=data_transform, train=False)
    return data
```

To adapt this model to a different dataset, change the `torchvision.datasets.FashionMNIST` to another dataset like `torchvision.datasets.CIFAR10` or your custom dataset.

## Notes:
- This implementation is primarily designed for experimentation with the **Fashion-MNIST** and **CIFAR-10** datasets and can be easily adapted to other datasets or use cases.
- The classifier guidance weight (`--guidance`) might need tuning for different datasets or classes.
  
Feel free to explore and modify the code to suit your needs!

## Examples FashionMNIST-Dataset:

| ![diffusion-1](https://github.com/user-attachments/assets/2fe5770a-d402-4865-8f0c-ec977c7ff20e) | ![diffusion-2](https://github.com/user-attachments/assets/7ee2d08b-15b1-4253-9ff4-a550df7c9668) | ![diffusion-3](https://github.com/user-attachments/assets/ce6ba0b8-33e2-4951-b1c5-90851d9b92c3) |
|:--:|:--:|:--:|
| ![diffusion-4](https://github.com/user-attachments/assets/25eb0b8b-9051-4af9-9518-7436f973d57b) | ![diffusion-5](https://github.com/user-attachments/assets/aa774bb4-597a-46e0-a91e-720c223eb90b) | ![diffusion](https://github.com/user-attachments/assets/e59f2e20-3b2d-472e-af3a-2744d7f7a332) |

## Examples CIFAR10-Dataset:

| ![image](https://github.com/user-attachments/assets/9fd476e4-dd39-42f6-b04f-d0a5d10651a6)| ![image](https://github.com/user-attachments/assets/b3e6e7d1-4c8d-46f0-b078-bd521a0c8578) | ![image](https://github.com/user-attachments/assets/5e3f97dc-6de7-410f-8d04-59947922fe7b)|
|:--:|:--:|:--:|
|'car'|'dog'|'ship'|



