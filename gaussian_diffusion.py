import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

#1D 배열에서 특정 timesteps에 해당하는 값을 추출하여 텐서로 변환하고, 지정된 모양으로 확장하는 함수
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr.to(device=timesteps.device)[timesteps].float() #[timesteps,] #timesteps에 해당하는 값들만 추출 -> (batch_size,)
    while len(res.shape) < len(broadcast_shape):
        res = res[...,None] #res의 차원을 늘려서 broadcast_shape에 맞게 확장 -> (batch_size, 1, 1)이 보통 경우
    
    return res.expand(broadcast_shape)

class GaussianDiffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()
        #beta to alpha 설정
        self.betas = betas #β_t (1000, ) 구조로 [0.0001 , ... , 0.02]
        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas 
        self.alphas_cumprod = np.cumprod(alphas, axis=0) 
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) 

        #q_sample을 위한 alpha 계산
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) #Training의 loss 함수에서 사용할  root(α_tilde), 해당 단계까지 가기 위한 노이즈 총량?
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) #Training의 loss 함수에서 사용할 root(1 - α_tilde)

        #p_sample을 위한 계산, 답지를 위한 q에 대한 변수 설정 포함
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        self.posterior_log_variance_clipped = np.log(np.maximum(self.posterior_variance, 1e-20))

        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) 

        self.posterior_mean_coef2 = np.sqrt(alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm_alphas_cumprod = np.sqrt((1.0 / self.alphas_cumprod) - 1.0)

    def _predict_xstart_from_eps(self, x_t, t, eps): #p_sample 식을 통해 x_0 도출, x_t, eps: (batch_size, seq_len, input_feats)
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            _extract_into_tensor(self.sqrt_recipm_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def q_sample(self, x_start, t, noise = None): #q_sample 식 (한번에 가는 방식)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def training_losses(self, model, x_start, t, noise=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start) #[batch_size, seq_len, input_feats]
        
        x_t = self.q_sample(x_start, t, noise) #[batch_size, seq_len, input_feats]

        model_output = model(x_t, t, **model_kwargs) #noise 예측
        target = noise

        with torch.no_grad(): # .detach()와 유사, 이 블록은 그래디언트 흐름에 영향을 주지 않음
            loss_root = F.mse_loss(model_output[:,:,:4], target[:,:,:4])
            loss_joint = F.mse_loss(model_output[:,:,4:], target[:,:,4:])
        
        final_loss = F.mse_loss(model_output, target)
        #weights = torch.exp(-2 * model.log_sigma)

        #final_loss = (loss_root * weights[0] + loss_joint * weights[1]) + model.log_sigma.sum()
        #model.log_sigma.data.clamp_(-5.0, 5.0) #안정화
        
        #with torch.no_grad(): # .detach()와 유사, 이 블록은 그래디언트 흐름에 영향을 주지 않음
        #    loss_for_logging = ((weights[0] * loss_root + weights[1] * loss_joint) / weights.sum())
        
        return {
            'loss': final_loss,
            'loss_root': loss_root.detach(),
            'loss_joint': loss_joint.detach(),
        }

    def p_mean_variance(self, model, x_t, t, model_kwargs=None): #모델을 통해 노이즈 예측하고 예측값으로부터 x_0을 구하고, x_{t-1}의 평균과 분산을 계산
        if model_kwargs is None:
            model_kwargs = {}

        model_output = model(x_t, t, **model_kwargs)
        pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
        model_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}
    
    def p_sample(self, model, x_t, t, model_kwargs=None): #위에서 구한 x_{t-1}의 평균과 분산을 통해 실제 샘플링을 수행
        if model_kwargs is None:
            model_kwargs = {}

        output = self.p_mean_variance(model, x_t, t, model_kwargs=model_kwargs)
        noise = torch.randn_like(x_t) 
        
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) #t가 0이냐 아니냐

        sample = output['mean'] + nonzero_mask * torch.exp(0.5 * output['log_variance']) * noise 
        return {'sample': sample, 'pred_xstart': output['pred_xstart']}
    
    def p_sample_loop(self, model, shape, model_kwargs=None): #샘플링 루프
        if model_kwargs is None:
            model_kwargs = {}

        device = next(model.parameters()).device
        
        motion = torch.randn(*shape, device=device) #초기 노이즈
        for i in tqdm(reversed(range(0,self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device) #현재 timestep
            motion = self.p_sample(model, motion, t, model_kwargs=model_kwargs)['sample']
        return motion