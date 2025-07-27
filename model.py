import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #[5000, 512] 각 위치를 512 dim 벡터로 표현
        position = torch.arange(0, max_len).unsqueeze(1) #[0,1,2,...,4999]인 1차원 텐서에 unsqueeze를 통해 -> [5000, 1] 크기의 2차원 텐서로 만듦
        #position = [[0,],[1,]...[4999,]] #인코딩할 위치 번호
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)) #??
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
        x = x + self.pe[:x.size(0), :] #[주어진 seq_len, 1, 512] ex)[196, 1, 512]
        return self.dropout(x)

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

class InputProcess(nn.Module):
    def __init__(self,input_feats, latent_dim):
        super().__init__()
        self.embedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2) #[seq_len, batch_size, input_feats]
        x = self.embedding(x)  # [seq_len, batch_size, latent_dim]
        return x
    
class OutputProcess(nn.Module):
    def __init__(self, latent_dim, output_feats):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_feats)

    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, output_feats]
        return x

class MotionTransformer(nn.Module):
    def __init__(self, njoints, nfeats, seq_len = None,
                latent_dim=256, ff_size = 1024, num_layers=8,
                num_heads = 4, dropout=0.1,**kargs):
        super().__init__()

        self.njoints = njoints #Number of joints in the skeleton : 23
        self.nfeats = nfeats #Number of features per joint : 6

        input_feats = njoints * nfeats + 4 #Input feature size (e.g., 23 joints * 6 features = 138) + 4

        self.latent_dim = latent_dim #Embedding dimension
        self.ff_size = ff_size #Feedforward size
        self.num_layers = num_layers #Number of transformer layers
        self.num_heads = num_heads #Number of attention heads
        self.dropout = dropout #Dropout rate

        self.input_process = InputProcess(input_feats, latent_dim) #입력 처리 레이어

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        self.embed_timestep = TimestepEmbedder(latent_dim, self.pos_encoder)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu'
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=None  # TransformerEncoderLayer에서 norm을 사용하지 않음
        )

        self.output_process = OutputProcess(latent_dim, input_feats)
          # 출력 처리 레이어
    
    def forward(self, x, timesteps):
        x_emb = self.input_process(x)  # [seq_len, batch_size, latent_dim]

        time_emb = self.embed_timestep(timesteps) # [1, batch_size, latent_dim]

        xseq = torch.cat((time_emb, x_emb), axis = 0)  # [seq_len + 1, batch_size, latent_dim]
        
        xseq = self.pos_encoder(xseq)

        output = self.seqTransEncoder(xseq)  # [seq_len + 1, batch_size, latent_dim]

        output = output[1:]

        predicted_noise = self.output_process(output)  # [batch_size, seq_len, njoints * nfeats]
        
        return predicted_noise