import torch.nn.functional as F
import torch
from tqdm.auto import tqdm


class ForwardDiffusion:

    def __init__(self, timesteps: int = 200, start: float =0.0001, end: float =0.02, scedule_type: str = "linear"  ) -> None:
        
        self.timesteps = timesteps
        self.start = start
        self.end = end

        if scedule_type == "linear":
            self.betas = self.linear_beta_schedule()
        elif scedule_type == "quadratic":
            self.betas = self.quadratic_beta_schedule()
        elif scedule_type == "sigmoid":
            self.betas = self.sigmoid_beta_schedule()
        else:
            self.betas = self.cosine_beta_schedule()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

 

        self.betas = self.betas.to(self.device)


        self.alphas = 1. - self.betas # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device) # (T,) calculate the cumulative product of the alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device) # (T,) calculate the square root of the cumulative product of the alphas
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device) # (T,) calculate 1 - the cumulative product of the alphas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(self.device)

        
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.

        If t.shape[0] != vals.shape[0], the function will return the sample for the corresponding index
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.to(self.device)).to(self.device) # t is a tensor of shape (batch_size,), returns a tensor of shape (batch_size,) with corresponding values

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)
    

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):

        x_start = x_start.to(self.device)
        t = t.to(self.device)

        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        else:
            noise = noise.to(self.device)

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        return torch.linspace(self.start, self.end, self.timesteps)

    def quadratic_beta_schedule(self):

        return torch.linspace(self.start**0.5, self.end**0.5, self.timesteps) ** 2

    def sigmoid_beta_schedule(self):
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (self.end - self.start) + self.start
    

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_, guidance):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        #pred_noise = (1+w) * model(x, t, class_) - w * model(x, t, torch.tensor([10]))
        pred_noise = model(x, t,torch.tensor([10])) + guidance * ((model(x, t, class_) - model(x, t,torch.tensor([10]))))

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_, guidance):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device).to(self.device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, class_, guidance)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, class_=None, guidance=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), class_=class_, guidance=guidance)   