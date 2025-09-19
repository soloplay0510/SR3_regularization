import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from regularization import *
from stdrelu import STDReLu,STDLeakyReLu,STDSLReLU
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        tv1_weight=None,
        tv2_weight=None,
        tvf_weight=None,
        tvf_alpha=1.6,
        wavelet_l1_weight = None,
        wavelet_type = "haar",
        std_activation_type="swish",
        std_normalize = True,
        nb_iterations=10,
        nb_kerhalfsize=1,
        leaky_alpha=0.2,
        sleaky_beta = 10.0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.tv1_weight = tv1_weight
        self.tv2_weight = tv2_weight
        self.tvf_weight = tvf_weight
        self.tvf_alpha = tvf_alpha
        self.wavelet_type = wavelet_type
        self.wavelet_l1_weight = wavelet_l1_weight
        self.std_activation_type = std_activation_type
        self.normalize = std_normalize
        self.nb_iterations = nb_iterations
        self.nb_kerhalfsize = nb_kerhalfsize
        self.leaky_alpha = leaky_alpha
        self.sleaky_beta = sleaky_beta    
        self.activation_map = {
            "identity":     lambda dim: nn.Identity(),
            "relu":         lambda dim: nn.ReLU(inplace=False),  # no duplicate!
            "tanh":         lambda dim: nn.Tanh(),
            "sigmoid":      lambda dim: nn.Sigmoid(),
            "softplus":     lambda dim: nn.Softplus(),
            "swish":        lambda dim: Swish(),
            "stdrelu":      lambda dim: STDReLu(
                n_channel=dim,
                nb_iterations=self.nb_iterations,
                nb_kerhalfsize=self.nb_kerhalfsize
            ),
            "stdleakyrelu": lambda dim: STDLeakyReLu(
                n_channel=dim,
                nb_iterations=self.nb_iterations,
                nb_kerhalfsize=self.nb_kerhalfsize,
                alpha=self.leaky_alpha
            ),
            "s_stdleakyrelu": lambda dim: STDSLReLU(
                n_channel=dim,
                nb_iterations=self.nb_iterations,
                nb_kerhalfsize=self.nb_kerhalfsize,
                alpha=self.leaky_alpha,
                beta=self.sleaky_beta
            ),
            "leakyrelu": lambda dim: nn.LeakyReLU(negative_slope=self.leaky_alpha, inplace=False),
        }
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        y_recon = self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
        return y_recon
   
    
    def predict_start_from_noise_pred(self, x_0, noise,noise_pred,c):
        x64 = x_0.to(torch.float64)
        n64 = noise.to(torch.float64)
        p64 = noise_pred.to(torch.float64)
        c64 = c.to(torch.float64)
        # clamp to avoid tiny negative under sqrt from roundoff
        s64 = torch.sqrt((1.0 - c64*c64).clamp_min(0.0))

        # guard division by near-zero c
        c_safe = c64.clamp_min(10.0 * torch.finfo(torch.float64).eps)

        y64 = x64 + (n64 - p64) * (s64 / c_safe)
        y = y64.to(x_0.dtype)
        # return x_0+(noise-noise_pred)*torch.sqrt(1.0/continuous_sqrt_alpha_cumprod**2-1.0)
        if self.normalize:
            y =  torch.sigmoid(y).clone()
        # map to [-1,1] instead of [0,1] if needed:
        # y = y * 2 - 1


        dim = y.shape[1]
    # apply activation if available
        if self.std_activation_type in self.activation_map:
            act = self.activation_map[self.std_activation_type](dim)
            y = act(y)
        return y

       
        
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None,generator = None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if t > 0 else torch.zeros_like(x)

        # noise = torch.randn_like(x,generator=generator) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False,noise=None, generator=None):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            if noise is None:
                img = torch.randn(shape, device=device,generator=generator)
            else:
                img = noise
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i,generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            if noise is None:
                img = torch.randn(shape, device=device,generator=generator)
            else:
                img = noise
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x,generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
           
        if continous:
            return ret_img
        else:
           
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False,generator=None):
        return self.p_sample_loop(x_in, continous,generator=generator)

    def predict_start_from_noise_continous(self, x_t,continuous_sqrt_alpha_cumprod, noise):
        x64 = x_t.to(torch.float64)
        c64 = continuous_sqrt_alpha_cumprod.to(torch.float64)
        n64 = noise.to(torch.float64)

        # Clamp to avoid sqrt of small negative due to fp rounding
        one_minus_c2 = (1.0 - c64 * c64).clamp_min(0.0)

        # Guard division by near-zero c
        c_safe = c64.clamp_min(10.0 * torch.finfo(torch.float64).eps)

        # Stable inverse: x0 = (x_t - sqrt(1 - c^2) * eps) / c
        x0_64 = (x64 - torch.sqrt(one_minus_c2) * n64) / c_safe

        return x0_64.to(x_t.dtype)
    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        # print('sqrt_alphas_cumprod_prev', self.sqrt_alphas_cumprod_prev.shape)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        y_recon = self.predict_start_from_noise_pred( x_start, noise,x_recon, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
        
        loss_noise = self.loss_func(x_start, y_recon)
        loss_TV1 = self.tv1_weight*TV1(y_recon)
        loss_TV2 = self.tv2_weight*TV2(y_recon)
        loss_TVF = self.tvf_weight*FTV(y_recon, alpha=self.tvf_alpha)
        loss_wave = self.wavelet_l1_weight*waveL1(y_recon, wname=self.wavelet_type)
        l_total = loss_noise + loss_TV1+ loss_TV2 + loss_TVF+ loss_wave
        return  {
            "total": l_total,
            "loss_noise": loss_noise,
            "loss_TV1": loss_TV1,
            "loss_TV2": loss_TV2,
            "loss_TVF": loss_TVF,
            "loss_wave_l1":loss_wave
        }


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
