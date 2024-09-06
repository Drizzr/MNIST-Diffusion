import torch.nn.functional as F
import torch



class Forward:

    def __init__(self, timesteps=50, start=0.0001, end=0.02) -> None:
        
        self.timesteps = timesteps
        self.betas = torch.linspace(start, end, timesteps) # (T,)
        self.alphas = 1. - self.betas # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # (T,) calculate the cumulative product of the alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # (T,) calculate the square root of the cumulative product of the alphas
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) # (T,) calculate 1 - the cumulative product of the alphas

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.

        If t.shape[0] != vals.shape[0], the function will return the sample for the corresponding index
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu()) # t is a tensor of shape (batch_size,), returns a tensor of shape (batch_size,) with corresponding values

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """

        noise = torch.randn_like(x_0) # sample noise from a normal distribution N(0, I)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


forward = Forward()