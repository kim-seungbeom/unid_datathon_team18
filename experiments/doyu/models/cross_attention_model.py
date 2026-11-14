"""원본 열화상 패치와 과거 센서 시퀀스를 결합하는 Cross-Attention 분류기.

Image: [B,1,64,64] -> CNN patch tokens [B,16,d]
Sensor: [B,T,5]     -> temporal tokens [B,T,d]
Cross attention: image tokens as Q, sensor tokens as K,V -> pooled -> 6 logits
"""
import torch
from torch import nn


class CrossAttentionNet(nn.Module):
    def __init__(self, sensor_dim=5, d_model=64, num_heads=4, dropout=0.15):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, d_model, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.sensor_proj = nn.Sequential(nn.Linear(sensor_dim,d_model),nn.GELU(),nn.LayerNorm(d_model))
        self.time_embed = nn.Parameter(torch.zeros(1,64,d_model))
        nn.init.normal_(self.time_embed,std=0.02)
        self.attention = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,6)
        )

    def forward(self, image, sequence):
        img=self.image_encoder(image).flatten(2).transpose(1,2)   # B,16,d
        seq=self.sensor_proj(sequence)
        if seq.shape[1]>self.time_embed.shape[1]:
            raise ValueError('Sequence length must be <= 64')
        seq=seq+self.time_embed[:,:seq.shape[1]]
        attended,_=self.attention(query=img,key=seq,value=seq,need_weights=False)
        fusion=self.norm(img+attended).mean(dim=1)
        pooled_sensor=seq.mean(dim=1)
        return self.classifier(torch.cat([fusion,pooled_sensor],dim=1))
