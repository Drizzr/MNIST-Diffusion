# MNIST-Diffusion

This repository contains a simple implementation of the Denoising Diffusion Probabilistic Model (DDPM) algorithm with classifier guidance. It is designed for training and sampling on the MNIST dataset, using a UNet architecture.

## Model Overview

- The **DDPM** algorithm follows the methodology described in the original DDPM paper.
- The **UNet architecture** is largely adapted from Hugging Face’s diffusion notebook: [annotated diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb).
- Classifier free guidance is implemented based on the approach in [this paper](https://arxiv.org/pdf/2207.12598).

## How to Sample

To sample from the trained model, run the following command:
```bash
python3 sample.py
```

### Sampling Options:
- `--c [0-9]`: Specify the class (0-9) for class-conditional sampling.
- `--w float`: Guidance strength. A lower `w` results in stronger guidance. The default value `w=0.8` seems to work well, though this may need adjustment for different classes.
- `--model_path`: Load a model checkpoint. Based on previous results, checkpoint from epoch 37 works best.
- `--batch_size`: Specify the number of images to sample. Ideally, this should be a square number for visualization purposes.
- `--animate`: Add this flag to visualize the reverse diffusion process for one random image in the batch.
- `--timesteps`: By default, the model is trained on 200 timesteps, but this can be adjusted during sampling.

## How to Train

To train the model, run the following command:
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
- `--load_optimizer`: Option to load the optimizer state from a checkpoint (default: True).
- `--clip`: Gradient clipping value (default: 10.0).
- `--lr`: Learning rate (default: 4e-4).
- `--save_dir`: Directory where model checkpoints will be loaded from.

## Notes:
- This implementation is primarily designed for experimentation with the MNIST dataset and should be easily adaptable for other datasets or use cases.
- The classifier guidance weight (`w`) might need tuning for different classes.
  
Feel free to explore and modify the code to suit your needs!

## Examples:

