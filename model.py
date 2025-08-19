import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
#x_0을 예측할 때 사용할 수 있는 고정적인 Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #[5000, 512] 각 위치를 512 dim 벡터로 표현
        position = torch.arange(0, max_len).unsqueeze(1) #[0,1,2,...,4999]인 1차원 텐서에 unsqueeze를 통해 -> [5000, 1] 크기의 2차원 텐서로 만듦
        #position = [[0,],[1,]...[4999,]] #인코딩할 위치 번호
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        #[0,2,4,...,510]에 아주 작은 음수값을 곱하고, 지수함수를 통해 [1,0.96,...,0.0001]과 같은 값을 만듦
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) #sin, cos 를 통해 위치 정보를 벡터로
        #position * div_term = [[0,1,2,...,4999], [1,0.96,0.95,...,0.0001]] -> [5000,256] #각 위치에 대한 sin/cos 값
        #torch.sin([5000,1] * [1,256]) -> torch.sin([5000,256]) -> [5000,256] #각 위치에 대한 sin 값
        pe = pe.unsqueeze(0).transpose(0, 1)
        # [1,5000,512]을 만들고, 그걸 transpose하여 [5000,1,512]로 만듦
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x의 shape는 [seq_len, batch_size, d_model]로 가정
        pos_embed = self.pe[:x.size(0), :]
        x = x + pos_embed.expand_as(x) 
        return self.dropout(x)
"""
'''
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder #positional encoding 모듈 그대로 가져옴

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),  # Timestep을 1차원으로 입력받음
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, timesteps):
        time_vector_from_pe = self.sequence_pos_encoder.pe[timesteps] #[batch_size, 1, latent_dim]
        embedded_time = self.time_embed(time_vector_from_pe)  # [batch_size, 1, latent_dim]
        return embedded_time.permute(1, 0, 2) #[1, batch_size, latent_dim] 형태로 변환하여 Transformer에 입력할 수 있도록 함
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 정보를 저장하고 학습할 수 있는 임베딩 레이어 생성
        self.embedding = nn.Embedding(max_len, d_model)
        # 임베딩 가중치를 안정적으로 초기화
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        
        # 현재 시퀀스 길이에 맞는 위치 인덱스 생성 (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        
        # 해당 위치의 학습된 임베딩 벡터를 가져옴. shape: [seq_len, d_model]
        pos_embed = self.embedding(positions)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [seq_len, 1, d_model]

        x = x + pos_embed

        return self.dropout(x)

def timestep_embedding(t, dim, max_period=10000):
    """
    시간(t) 정보를 sin/cos 함수를 이용해 벡터로 변환합니다.
    t: (Batch Size)
    dim: 임베딩 벡터의 차원 (latent_dim)
    """
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class InputProcess(nn.Module):
    def __init__(self,input_feats, latent_dim):
        super().__init__()
        self.embedding = nn.Linear(input_feats, latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.permute(1, 0, 2) #[seq_len, batch_size, input_feats]
        x = self.embedding(x)  #[seq_len, batch_size, latent_dim]
        return x
    
class OutputProcess(nn.Module):
    def __init__(self, latent_dim, output_feats):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_feats)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, output_feats]
        return x

class MotionTransformer(nn.Module):
    def __init__(self, input_feats, seq_len = None,
                latent_dim=512, ff_size=3072, num_layers=12,
                num_heads = 8, dropout=0.1,
                time_integration_method='concat',
                **kargs):
        super().__init__()

        self.input_feats = input_feats #입력 특징 수
        self.latent_dim = latent_dim #Embedding dimension
        self.ff_size = ff_size #Feedforward size
        self.num_layers = num_layers #Number of transformer layers
        self.num_heads = num_heads #Number of attention heads
        self.dropout = dropout #Dropout rate
        self.norm = nn.LayerNorm(latent_dim) #Layer normalization
        
        self.time_integration_method = time_integration_method #Time integration method

        self.input_process = InputProcess(self.input_feats, self.latent_dim) #입력 처리 레이어

        self.pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first = False
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=self.norm
        )

        self.output_process = OutputProcess(latent_dim, input_feats)
          # 출력 처리 레이어
    
    def forward(self, x, timesteps):
        '''
        x_emb = self.input_process(x)
        seq_len = x_emb.size(0)
        batch_size = x_emb.size(1)

        time_emb = self.embed_timestep(timesteps)
        time_emb = time_emb.squeeze(0)
        time_emb = time_emb.unsqueeze(0).expand(seq_len, -1, -1)
        x_emb = x_emb + time_emb

        x_emb = self.pos_encoder(x_emb)

        encoded = self.seqTransEncoder(x_emb) 

        predicted_noise = self.output_process(encoded)
        '''
  
        x_emb = self.input_process(x)  # [seq_len, batch_size, latent_dim]

        time_emb_sin = timestep_embedding(timesteps, self.latent_dim) # [batch_size, latent_dim]
        time_emb = self.time_mlp(time_emb_sin) #[batch_size, latent_dim]
        if self.time_integration_method == 'concat':
            time_emb_token = time_emb.unsqueeze(0) # [1, batch_size, latent_dim]
            x_seq = torch.cat((time_emb_token, x_emb), axis=0) #[seq_len + 1, batch_size, latent_dim]
        elif self.time_integration_method == 'add':
            # 시간 임베딩을 모든 프레임에 더해주기 위해 확장(expand)
            time_emb_broadcast = time_emb.unsqueeze(0).expand_as(x_emb)
            x_seq = x_emb + time_emb_broadcast
        else:
            raise ValueError(f"Unknown time integration method: {self.time_integration_method}")

        x_seq = self.pos_encoder(x_seq) #[seq_len + 1, batch_size, latent_dim]

        output = self.seqTransEncoder(x_seq)  # [seq_len + 1, batch_size, latent_dim]

        if self.time_integration_method == 'concat':
            output = output[1:] # concat 방식일 때만 시간 토큰에 해당하는 출력을 제거

        predicted_noise = self.output_process(output)  # [batch_size, seq_len, input_feats]

        return predicted_noise