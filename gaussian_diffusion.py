#gaussian_diffusion.py
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
            loss_joint = F.mse_loss(model_output[:,:,4:208], target[:,:,4:208])
            loss_foot = F.mse_loss(model_output[:,:,208:210], target[:,:,208:210])
        
        final_loss = F.mse_loss(model_output, target)
        
        return {
            'loss': final_loss,
            'loss_root': loss_root.detach(),
            'loss_joint': loss_joint.detach(),
            'loss_foot': loss_foot.detach(),
        }

    def p_mean_variance(self, model_output, x_t, t): #모델을 통해 노이즈 예측하고 예측값으로부터 x_0을 구하고, x_{t-1}의 평균과 분산을 계산
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

        model_output = model(x_t, t, **model_kwargs)
        output = self.p_mean_variance(model_output, x_t, t)
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
    
    def cfg_p_sample_loop(self, model, shape, model_kwargs=None, guidance_scale=3.0):
        if model_kwargs is None:
            model_kwargs = {}

        if guidance_scale <= 1.0 or (model_kwargs.get('classes_name') is None and model_kwargs.get('classes_type') is None):
            print("Warning: guidance_scale <= 1.0 or no class provided, running standard p_sample_loop.")
            return self.p_sample_loop(model, shape, model_kwargs=model_kwargs)
        
        device = next(model.parameters()).device
        
        motion = torch.randn(*shape, device=device) 
        for i in tqdm(reversed(range(0,self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device) 
            cond_model_output = model(motion, t, **model_kwargs)

            uncond_kwargs = model_kwargs.copy()
            uncond_kwargs['classes_name'] = None
            uncond_kwargs['classes_type'] = None
            uncond_model_output = model(motion, t, **uncond_kwargs)

            guided_model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)
            
            output = self.p_mean_variance(guided_model_output, motion, t)
            
            model_mean = output['mean']
            model_log_variance = output['log_variance']

            noise = torch.randn_like(motion) if i > 0 else 0
            motion = model_mean + torch.exp(0.5 * model_log_variance) * noise
        
        return motion
    
    #####################################################################################################################

    def q_sample_traj(self, x_start, t, noise = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return x_t
    
    def training_losses_traj(self, model, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start) #[batch_size, seq_len, input_feats]

        x_t = self.q_sample_traj(x_start, t, noise) #[batch_size, seq_len, input_feats]

        model_output = model(x_t, t) #noise 예측

        target = noise

        with torch.no_grad(): # .detach()와 유사, 이 블록은 그래디언트 흐름에 영향을 주지 않음
            loss_root = F.mse_loss(model_output[:,:,:4], target[:,:,:4])
            loss_joint = F.mse_loss(model_output[:,:,4:208], target[:,:,4:208])
            loss_foot = F.mse_loss(model_output[:,:,208:210], target[:,:,208:210])
            loss_cond = F.mse_loss(model_output[:,:,210:213], target[:,:,210:213])
        
        final_loss = F.mse_loss(model_output, target)

        return {
            'loss': final_loss,
            'loss_root': loss_root.detach(),
            'loss_joint': loss_joint.detach(),
            'loss_foot': loss_foot.detach(),
            'loss_cond': loss_cond.detach(),
        }

    def p_sample_traj(self, model, x_t, t): #위에서 구한 x_{t-1}의 평균과 분산을 통해 실제 샘플링을 수행
        
        model_output = model(x_t, t)

        output = self.p_mean_variance(model_output, x_t, t)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) #t가 0이냐 아니냐

        sample = output['mean'] + nonzero_mask * torch.exp(0.5 * output['log_variance']) * noise 
                
        return {'sample': sample, 'pred_xstart': output['pred_xstart']}

    def p_sample_loop_traj(self, model, shape, traj_norm): #샘플링 루프

        device = next(model.parameters()).device
        B, T, F = shape
        assert traj_norm.shape == (1, T, 3), f"traj_norm shape {traj_norm.shape} != (1,{T},3)"
        traj_norm = traj_norm.to(device)
        
        motion = torch.randn(*shape, device=device) #초기 노이즈

        for i in tqdm(reversed(range(0,self.num_timesteps)), desc='Traj sampling loop time step', total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device) #현재 timestep
            def extract(arr):
                return _extract_into_tensor(arr, t, (B, T, 3))

            sqrt_alpha_t     = extract(self.sqrt_alphas_cumprod)
            sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod)

            # cond 입력 구성: 원하는 traj의 x_t = √α_t · x0 + √(1-α_t) · ε
            # ε는 새로 뽑아도 되고(분포만 맞으면 됨), 고정해도 됨
            eps_cond     = torch.randn(B, T, 3, device=device)
            traj_cond_xt = sqrt_alpha_t * traj_norm + sqrt_one_minus_t * eps_cond

            # 모델 입력 (cond) 주입
            motion_in = motion.clone()
            motion_in[:, :, 210:213] = traj_cond_xt

            eps = model(motion_in, t)   # cond 입력으로 예측

            out = self.p_mean_variance(eps, motion, t)
            noise = torch.randn_like(motion) if i > 0 else 0
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(shape)-1)))
            motion = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return motion
        
    def cfg_p_sample_loop_traj(self, model, shape, traj_norm, guidance_scale: float = 3.0,):
        if guidance_scale <= 1.0:
            return self.p_sample_loop_traj(model, shape, traj_norm)
        
        device = next(model.parameters()).device
        B, T, F = shape
        assert traj_norm.shape == (1, T, 3)
        traj_norm = traj_norm.to(device)

        # 1.  초기 노이즈
        motion = torch.randn(*shape, device=device)

        for i in tqdm( reversed(range(self.num_timesteps)), desc="CFG-Inpaint sampling", total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device)
            
            def extract(scalar_arr):
                return _extract_into_tensor(scalar_arr, t, (B, T, 3))

            sqrt_alpha_t      = extract(self.sqrt_alphas_cumprod)
            sqrt_one_minus_t  = extract(self.sqrt_one_minus_alphas_cumprod)

            # ─ cond 입력 구성: 원하는 traj의 x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*eps
            #   eps는 fresh noise로 괜찮다. (분포만 맞으면 OK)
            eps_cond = torch.randn(B, T, 3, device=device)
            traj_cond_xt = sqrt_alpha_t * traj_norm + sqrt_one_minus_t * eps_cond

            motion_cond = motion.clone()
            motion_cond[:, :, 210:213] = traj_cond_xt  # cond: 원하는 traj 주입
            cond_eps = model(motion_cond, t)

            # ─ uncond 입력 구성: x_start=0 → x_t = sqrt(1-a_t)*eps
            eps_uncond = torch.randn(B, T, 3, device=device)
            traj_uncond_xt = sqrt_one_minus_t * eps_uncond

            motion_uncond = motion.clone()
            motion_uncond[:, :, 210:213] = traj_uncond_xt
            uncond_eps = model(motion_uncond, t)

            # ─ CFG
            guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

            # ─ p_mean_variance → x_{t-1}
            out = self.p_mean_variance(guided_eps, motion, t)
            noise = torch.randn_like(motion) if i > 0 else 0
            nonzero_mask = (t != 0).float().view(-1, *([1]*(len(shape)-1)))
            motion = out['mean'] + nonzero_mask * torch.exp(0.5*out['log_variance']) * noise
        return motion
    

    ######################################################################################################################

    def training_losses_inpaint(self, model, x_start, t, traj_cond, traj_prob, noise=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start) #[batch_size, seq_len, input_feats]

        x_t = self.q_sample(x_start, t, noise) #[batch_size, seq_len, input_feats]

        if traj_prob > 0.0:
            drop_mask = (torch.rand(x_start.size(0), device=x_start.device) < traj_prob).view(-1, 1, 1)
            traj_cond_in = torch.where(drop_mask, torch.zeros_like(traj_cond), traj_cond)
        else:
            traj_cond_in = traj_cond

        model_output = model(x_t, traj_cond_in, t, **model_kwargs) #noise 예측

        target = noise

        with torch.no_grad(): # .detach()와 유사, 이 블록은 그래디언트 흐름에 영향을 주지 않음
            loss_root = F.mse_loss(model_output[:,:,:4], target[:,:,:4])
            loss_joint = F.mse_loss(model_output[:,:,4:208], target[:,:,4:208])
            loss_foot = F.mse_loss(model_output[:,:,208:210], target[:,:,208:210])

        final_loss = F.mse_loss(model_output, target)

        return {
            'loss': final_loss,
            'loss_root': loss_root.detach(),
            'loss_joint': loss_joint.detach(),
            'loss_foot': loss_foot.detach(),
        }

    def p_sample_inpaint(self, model, x_t, t, traj_cond, model_kwargs = None): #위에서 구한 x_{t-1}의 평균과 분산을 통해 실제 샘플링을 수행
        if model_kwargs is None:
            model_kwargs = {}

        model_output = model(x_t, traj_cond, t, **model_kwargs)

        output = self.p_mean_variance(model_output, x_t, t,)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) #t가 0이냐 아니냐

        sample = output['mean'] + nonzero_mask * torch.exp(0.5 * output['log_variance']) * noise 
                
        return {'sample': sample, 'pred_xstart': output['pred_xstart']}

    def p_sample_loop_inpaint(self, model, shape, traj_cond, model_kwargs=None): #샘플링 루프
        if model_kwargs is None:
            model_kwargs = {}

        device = next(model.parameters()).device
        
        motion = torch.randn(*shape, device=device) #초기 노이즈

        for i in tqdm(reversed(range(0,self.num_timesteps)), desc='Inpaint sampling loop time step', total=self.num_timesteps):
            t = torch.tensor([i] * shape[0], device=device) #현재 timestep
            motion = self.p_sample_inpaint(model, motion, t, traj_cond, model_kwargs)['sample']

        return motion

    def cfg_p_sample_loop_inpaint(self, model, shape, traj_cond, guidance_scale: float = 3.0, model_kwargs=None):
        """
        shape        : (batch, seq_len, feat=213)
        x_condition  : [B,T,213]  –  trajectory 가 포함된 조건 텐서
        inpaint_mask : [B,T,213]  –  trajectory 부분 True
        guidance_scale > 1.0  →  CFG 사용
        """
        if guidance_scale <= 1.0:
            return self.p_sample_loop_inpaint(model, shape, traj_cond, model_kwargs=model_kwargs)
        
        if model_kwargs is None:
            model_kwargs = {}

        device = next(model.parameters()).device
        B, T, F = shape

        # 1.  초기 노이즈
        motion = torch.randn(*shape, device=device)
        
        zero_traj = torch.zeros_like(traj_cond, device=device)

        for i in tqdm(
            reversed(range(self.num_timesteps)),
            desc="CFG-Inpaint sampling", total=self.num_timesteps
        ):
            t = torch.full((B,), i, device=device, dtype=torch.long)
        
            if guidance_scale <= 1.0:
                # No CFG
                pred_noise = model(motion, traj_cond, t, **model_kwargs)
            else:
                # CFG: conditional + unconditional
                # Conditional (with trajectory)
                eps_cond = model(motion, traj_cond, t, **model_kwargs)
                
                # Unconditional (without trajectory)
                uncond_kwargs = model_kwargs.copy()
                uncond_kwargs['classes_name'] = None  # class도 제거

                eps_uncond = model(motion, zero_traj, t, **uncond_kwargs)

                # CFG
                pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # p_mean_variance
            output = self.p_mean_variance(pred_noise, motion, t)
            
            # Sampling
            noise = torch.randn_like(motion) if i > 0 else 0
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            motion = output['mean'] + nonzero_mask * torch.exp(0.5 * output['log_variance']) * noise
        
        return motion    
    ######################################################################################################################

    def training_losses_cond(self, model, x_start, t, cond, cond_drop_prob: float = 0.1, model_kwargs=None):
        """
        cond를 별도 입력으로 쓰는 학습 손실(= 표준 DDPM + 간단 CFG 드롭).
        x_start: [B,T,210] (정규화)
        cond:    [B,T,3]   (정규화된 vx,vz,yaw_rate)
        """

        if model_kwargs is None:
            model_kwargs = {}
        
        target = torch.randn_like(x_start) # target ε ~ N(0,I)
        x_t = self.q_sample(x_start, t, target) # x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

        # classifier-free: cond 일부 드롭
        if cond_drop_prob > 0.0:
            drop = (torch.rand(x_start.size(0), device=x_start.device) < cond_drop_prob).view(-1,1,1) # [Batch, 1, 1]
            cond_in = torch.where(drop, torch.zeros_like(cond), cond) # drop[b] == True -> cond 0으로
        else:
            cond_in = cond

        model_output = model(x_t, t, cond_in, **model_kwargs)  # ← 모델이 cond 인자를 받도록만 해주면 됨
        loss = F.mse_loss(model_output, target)

        with torch.no_grad(): # .detach()와 유사, 이 블록은 그래디언트 흐름에 영향을 주지 않음
            loss_root = F.mse_loss(model_output[:,:,:4], target[:,:,:4])
            loss_joint = F.mse_loss(model_output[:,:,4:208], target[:,:,4:208])
            loss_foot = F.mse_loss(model_output[:,:,208:210], target[:,:,208:210])
        
        return {
            'loss': loss,
            'loss_root': loss_root.detach(),
            'loss_joint': loss_joint.detach(),
            'loss_foot': loss_foot.detach(),
        }

    def p_sample_loop_cond(self, model, shape, cond, guidance_scale: float = 3.0, model_kwargs=None):
        """
        cond + CFG를 쓰는 샘플러. inpaint 불필요.
        shape: (B,T,210), cond: [B,T,3] (정규화)
        """

        if model_kwargs is None:
            model_kwargs = {}
    
        device = next(model.parameters()).device
        B, T, F = shape
        x = torch.randn(*shape, device=device)

        zero_cond = torch.zeros_like(cond, device=device)  # uncond 분기
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling loop time step', total=self.num_timesteps):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            eps_c = model(x, t, cond=cond, **model_kwargs)               # conditional
            if guidance_scale <= 1.0:
                eps = eps_c
            else:
                uncond_kwargs = model_kwargs.copy()
                uncond_kwargs['classes_name'] = None

                eps_u = model(x, t, cond=zero_cond, **uncond_kwargs)      # unconditional 여기 model_kwargs를 어떻게?
                eps = eps_u + guidance_scale * (eps_c - eps_u)

            out = self.p_mean_variance(eps, x, t)
            noise = torch.randn_like(x) if i > 0 else 0
            nonzero = (t != 0).float().view(-1, *([1]*(len(shape)-1)))
            x = out['mean'] + nonzero * torch.exp(0.5*out['log_variance']) * noise

        return x
    
    def p_sample_loop_cond_overwrite(self, model, shape, cond, guidance_scale: float = 3.0, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        device = next(model.parameters()).device
        B, T, F = shape
        x = torch.randn(*shape, device=device)

        zero_cond = torch.zeros_like(cond, device=device)
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling loop time step', total=self.num_timesteps):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # 1) eps 예측 (CFG 포함)
            eps_c = model(x, t, cond=cond, **model_kwargs)
            if guidance_scale <= 1.0:
                eps = eps_c
            else:
                uncond_kwargs = model_kwargs.copy()
                uncond_kwargs['classes_name'] = None
                eps_u = model(x, t, cond=zero_cond, **uncond_kwargs)
                eps = eps_u + guidance_scale * (eps_c - eps_u)

            # 2) pred_xstart 복원
            pred_x0 = self._predict_xstart_from_eps(x, t, eps)   # [B,T,F], 정규화 공간의 x̂₀

            # 3) 🔒 하드 고정: 루트 velocity & yaw_rate 채널을 cond로 교체
            #    주의: cond는 이미 정규화된 [vx_local, vz_local, dyaw]여야 함
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, 1:4] = cond  # 1:4 == [vx, vz, dyaw]  (0은 root_y)

            # 4) posterior(mean/var) 재계산 (교체된 x̂₀ 기반)
            coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape)
            coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x.shape)
            model_mean = coef1 * pred_x0 + coef2 * x
            logvar = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

            noise = torch.randn_like(x) if i > 0 else 0
            nonzero = (t != 0).float().view(-1, *([1]*(len(shape)-1)))
            x = model_mean + nonzero * torch.exp(0.5 * logvar) * noise

        return x