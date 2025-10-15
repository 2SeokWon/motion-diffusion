#model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#x_0을 예측할 때 사용할 수 있는 고정적인 Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #[5000, 512] 각 위치를 512 dim 벡터로 표현
        position = torch.arange(0, max_len).unsqueeze(1) #[0,1,2,...,4999]인 1차원 텐서에 unsqueeze를 통해 -> [5000, 1] 크기의 2차원 텐서로 만듦
                                                         #position = [[0,],[1,]...[4999,]] #인코딩할 위치 번호
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)) #[0,2,4,...,510]에 아주 작은 음수값을 곱하고, 지수함수를 통해 [1,0.96,...,0.0001]과 같은 값을 만듦
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) #sin, cos 를 통해 위치 정보를 벡터로
                                                     #position * div_term = [[0,1,2,...,4999], [1,0.96,0.95,...,0.0001]] -> [5000,256] #각 위치에 대한 sin/cos 값
                                                     #torch.sin([5000,1] * [1,256]) -> torch.sin([5000,256]) -> [5000,256] #각 위치에 대한 sin 값
        pe = pe.unsqueeze(0).transpose(0, 1)         # [1,5000,512]을 만들고, 그걸 transpose하여 [5000,1,512]로 만듦
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x의 shape는 [seq_len, batch_size, d_model]로 가정
        pos_embed = self.pe[:x.size(0), :]
        x = x + pos_embed.expand_as(x) 
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

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embed_layer = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.embed_layer.weight, mean=0.0, std=0.02)

    def forward(self, classes):
        return self.embed_layer(classes)

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
                latent_dim=1024, ff_size=4096, num_layers=8,
                num_heads = 8, dropout=0.1,
                **kargs):
        super().__init__()

        self.input_feats = input_feats #입력 특징 수
        self.latent_dim = latent_dim #Embedding dimension
        self.ff_size = ff_size #Feedforward size
        self.num_layers = num_layers #Number of transformer layers
        self.num_heads = num_heads #Number of attention heads
        self.dropout = dropout #Dropout rate
        self.norm = nn.LayerNorm(latent_dim) #Layer normalization
        
        self.input_process = InputProcess(self.input_feats, self.latent_dim) #입력 처리 레이어

        self.pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.class_name_embedding = ClassEmbedding(num_classes=7, dim=latent_dim) # 클래스 이름 임베딩 레이어
        self.null_class_name_emb = nn.Parameter(torch.zeros(1, self.latent_dim)) # null token embedding

        self.class_type_embedding = ClassEmbedding(num_classes=7, dim=latent_dim) # 클래스 타입 임베딩 레이어
        self.null_class_type_emb = nn.Parameter(torch.zeros(1, self.latent_dim))

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
    
    def forward(self, x, timesteps, **model_kwargs):
        name_classes = model_kwargs.get('classes_name', None)
        type_classes = model_kwargs.get('classes_type', None)
        #if traj_mask:
        #    x[:,:,208:211] = 0.0 #조건부 생성 시, 조건부 특징을 0으로 마스킹
        
        x_emb = self.input_process(x)  # [seq_len, batch_size, latent_dim]

        time_emb_sin = timestep_embedding(timesteps, self.latent_dim) # [batch_size, latent_dim]
        time_emb = self.time_mlp(time_emb_sin) #[batch_size, latent_dim]
        time_emb_token = time_emb.unsqueeze(0) # [1, batch_size, latent_dim]

        if name_classes is None:
            batch_size = x_emb.size(1)
            class_emb = self.null_class_name_emb.expand(batch_size, -1) #[batch_size, latent_dim]
        else:
            class_emb = self.class_name_embedding(name_classes) #[batch_size, latent_dim]

        class_emb_token = class_emb.unsqueeze(0) # [1, batch_size, latent_dim]

        if type_classes is None:
            batch_size = x_emb.size(1)
            class_type_emb = self.null_class_type_emb.expand(batch_size, -1) #[batch_size, latent_dim]
        else:
            class_type_emb = self.class_type_embedding(type_classes) #[batch_size, latent_dim]
        
        class_type_emb_token = class_type_emb.unsqueeze(0) # [1, batch_size, latent_dim]

        x_seq = torch.cat((time_emb_token, class_emb_token, class_type_emb_token, x_emb), axis=0) #[seq_len + 3, batch_size, latent_dim]

        x_seq = self.pos_encoder(x_seq) #[seq_len + 3, batch_size, latent_dim]

        output = self.seqTransEncoder(x_seq)  # [seq_len + 3, batch_size, latent_dim]

        output = output[3:] # [seq_len, batch_size, latent_dim] (처음 세 토큰 제거)

        predicted_noise = self.output_process(output)  # [batch_size, seq_len, input_feats]

        return predicted_noise