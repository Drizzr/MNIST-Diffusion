from torch import nn
import math
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(emb_dim, out_ch)
        self.class_mlp = nn.Linear(emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, c):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t)) # -> (batch, out_ch)
        class_emb = self.relu(self.class_mlp(c)) # -> (batch, out_ch)
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2] # reshapes (batch, out_ch) -> (batch, out_ch, 1, 1)
        class_emb = class_emb[(..., ) + (None, ) * 2] # reshapes (batch, out_ch) -> (batch, out_ch, 1, 1)

        # Add time channel

        h = h + time_emb + class_emb # sum is broadcasted to (batch, out_dim, H, W)
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding



class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (128, 256, 512)
        up_channels = (512, 256, 128)
        out_dim = 1 
        emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU()
            )
        
        self.class_mlp = nn.Sequential(
                torch.nn.Embedding(11, emb_dim, padding_idx=10),  # 11 classes => 10 + 1 "empty" class
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, class_label):
        # x shape: (batch, 1, height, width)
        # timestep shape: (batch,)
        # class_label shape: (batch,)


        # Embedd time
        t = self.time_mlp(timestep) # -> (batch, time_emb_dim)

        # Embedd class
        c = self.class_mlp(class_label)

        # Initial conv
        x = self.conv0(x) # -> (batch, 28, height, width)

        # Unet
        residual_inputs = [] # Store inputs for skip connections
        for down in self.downs:
            x = down(x, t, c) # -> (batch, 2 * C', H'/2, W'/2) every iteration
            residual_inputs.append(x)
        
        # x shape: (batch, 112, 7, 7)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels

            x = x + residual_x         
            x = up(x, t, c) # -> (batch, C'/2, H*2, W*2) every iteration
        return self.output(x)
    

if __name__ == "__main__":
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))


    test_tensor = torch.randn(1, 1, 28, 28)

    # Test forward pass

    out = model(test_tensor, torch.tensor([1]), torch.tensor([1]))
    print(out.shape)