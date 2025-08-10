import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from kinematics import features_to_xyz

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

        alphas = 1.0 - self.betas #α_t = 1 - β_t 로 (1000 ,)
        self.alphas_cumprod = np.cumprod(alphas, axis=0) #α_t 누적곱, α_tilde (1000, )
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) #α_t-1 누적곱, α_tilde_prev (1000, ) #α_tilde_prev = [1.0, α_0, α_1, ..., α_998]

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

    def _predict_xstart_from_eps(self, x_t, t, eps): #q_sample 식을 통해 x_0 도출, x_t, eps: (batch_size, seq_len, input_feats)
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
    
    def training_losses(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)

        model_output = model(x_t, t) #이걸 노이즈로 나오게 했음 x_t에서 x_0으로 가는 노이즈?

        target = noise
        
        loss_simple = F.mse_loss(model_output, target)

        pred_x_start = self._predict_xstart_from_eps(x_t, t, model_output)
        target_velocity = x_start[:, 1:] - x_start[:, :-1] #x_start의 속도, (batch_size, seq_len-1, input_feats)
        pred_velocity = pred_x_start[:, 1:] - pred_x_start[:, :-1]
        loss_vel = F.mse_loss(pred_velocity, target_velocity)

        lambda_vel = 0.3 #속도에 대한 가중치
        
        #lambda_fk = 0.0 #가중치 변경 가능 당장은 사용 X
        #loss_fk = torch.tensor(0.0, device=x_start.device)

        #if lambda_fk > 0.0:
        #    pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
        #    target_xyz = features_to_xyz(x_start, skeleton, mean, std)
        #    pred_xyz = features_to_xyz(pred_xstart, skeleton, mean, std)

        #    loss_fk = F.mse_loss(pred_xyz, target_xyz)
        
        final_loss = loss_simple + lambda_vel * loss_vel
        
        final_loss = loss_simple #현재는 FK loss를 사용하지 않음
        return {
            'loss': final_loss, 
            'loss_simple': loss_simple.item(),
            'loss_vel': loss_vel.item()
        }

    def p_mean_variance(self, model, x_t, t): #모델을 통해 노이즈 예측하고 예측값으로부터 x_0을 구하고, x_{t-1}의 평균과 분산을 계산
        model_output = model(x_t, t)
        pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
        model_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}
    
    def p_sample(self, model, x_t, t): #위에서 구한 x_{t-1}의 평균과 분산을 통해 실제 샘플링을 수행

        output = self.p_mean_variance(model, x_t, t)
        noise = torch.randn_like(x_t) 
        
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) #t가 0이냐 아니냐

        sample = output['mean'] + nonzero_mask * torch.exp(0.5 * output['log_variance']) * noise 
        return {'sample': sample, 'pred_xstart': output['pred_xstart']}
    
    def p_sample_loop(self, model, shape): #샘플링 루프
        device = next(model.parameters()).device
        
        motion = torch.randn(*shape, device=device) #초기 노이즈
        for i in tqdm(reversed(range(0,self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device) #현재 timestep
            motion = self.p_sample(model, motion, t)['sample']

        return motion