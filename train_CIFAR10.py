import argparse
import torchvision
from model.u_net import Unet
from model.forward import ForwardDiffusion
from model.trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms 
import torch
import sys
import json
import os



def load_from_checkpoint(args, forward, dataset, val_dataset, writer):
    """Load model from checkpoint"""
    print("loading model from checkpoint...")
    with open(os.path.join(args.save_dir, "params.json"), "r") as f:
        params = json.load(f)

    model = Unet(
        dim=32,
        channels=3,
        dim_mults=(1, 2, 4,)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, step_size_down=None, 
                                mode='triangular2', gamma=0.1, scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pth"), weights_only=True))


    optimizer.load_state_dict(torch.load(os.path.join(args.save_dir, "optimizer.pth"), weights_only=True))
    lr_scheduler.load_state_dict(torch.load(os.path.join(args.save_dir, "lr_scheduler.pth"), weights_only=True))

    print("model loaded successfully...")

    trainer = Trainer(model, dataset, args, val_dataset, 
                        init_epoch=params["epoch"], 
                        last_epoch=args.num_epochs, writer=writer, optimizer=optimizer, forward_diffusion=forward, 
                        p_uncond=params["p_uncond"], timesteps=params["timesteps"],
                        lr_scheduler=lr_scheduler)

    current_batch_size = args.batch_size
    checkpoint_batch_size = params["batch_size"]


    trainer.step = params["step"] * (checkpoint_batch_size // current_batch_size)
    trainer.total_step = params["total_step"] 

    return trainer, model


def load_transformed_dataset(train=True):
    data_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    if train:
        data = torchvision.datasets.CIFAR10(root="./data_CIFAR10", download=True, 
                                            transform=data_transform, train=True)

    else:
        data = torchvision.datasets.CIFAR10(root="./data_CIFAR10", download=True, 
                                            transform=data_transform, train=False)
    return data

    

def main():

    # get args
    parser = argparse.ArgumentParser(description="Train the model.")

    # model args

    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--print_freq", type=int, default=100)
    
    parser.add_argument("--p_uncond", type=float, default=0.2, help="probability of unconditional sampling")

    parser.add_argument("--timesteps", type=int, default=200, help="number of timesteps")

    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")
    
    parser.add_argument("--clip", type=float, default=10.0, help="gradient clipping")

    parser.add_argument("--lr", type=float, default=4*10**(-4), help="learning rate")

    parser.add_argument("--save_dir", type=str, help="directory of the saved checkpoint of the model, also where the model will be saved", default="checkpoints_CIFAR10/")


    args = parser.parse_args()

    from_check_point = args.from_check_point
    
    
    print("_________________________________________________________________")
    print("HYPERPARAMETERS: ")
    for arg in vars(args):
        print(arg,": ", getattr(args, arg))
    print("_________________________________________________________________")
    

    dataset = DataLoader(load_transformed_dataset(train=True), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataset = DataLoader(load_transformed_dataset(train=False), batch_size=args.batch_size, shuffle=True, drop_last=True)


    print(len(dataset), len(val_dataset))
    writer = SummaryWriter("runs/") # tensorboard writer

    forward = ForwardDiffusion(timesteps=args.timesteps, start=0.0001, end=0.02)

    print("sucessfully loaded dataset...")

    if from_check_point:
        trainer, model = load_from_checkpoint(args, forward, dataset, val_dataset, writer)
    
    else:
        model = Unet(
            dim=28,
            channels=3,
            dim_mults=(1, 2, 4,)
            )
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, step_size_down=None, 
                                mode='triangular2', gamma=0.1, scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)


        trainer = Trainer(model, dataset, args, val_dataset, writer=writer, optimizer=optimizer, forward_diffusion=forward, timesteps=args.timesteps, p_uncond=args.p_uncond, lr_scheduler=lr_scheduler)
    
    print("_________________________________________________________________")
    print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("_________________________________________________________________")

    try:
        trainer.train()
        writer.close()
    except KeyboardInterrupt as e:
        print(e)
        
        trainer.save_model()

        writer.close()
        
        sys.exit()


if __name__ == "__main__":
    main()