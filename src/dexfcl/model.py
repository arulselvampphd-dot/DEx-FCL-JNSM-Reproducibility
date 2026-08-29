from __future__ import annotations
import torch
from torch import nn

class FeatureGate(nn.Module):
    def __init__(self,d:int,hidden:int|None=None):
        super().__init__(); h=hidden or max(16,min(128,d))
        self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,d))
    def forward(self,x):
        # Multiplication by d preserves the average input scale while retaining normalized gate weights.
        return torch.softmax(self.net(x),dim=-1)*x.shape[-1]

class DExNet(nn.Module):
    def __init__(self,input_dim:int,n_classes:int,hidden1:int=128,latent_dim:int=64,dropout:float=.2):
        super().__init__(); self.input_dim=input_dim; self.n_classes=n_classes; self.latent_dim=latent_dim
        self.gate=FeatureGate(input_dim)
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden1),nn.LayerNorm(hidden1),nn.ReLU(),nn.Dropout(dropout),
                                   nn.Linear(hidden1,latent_dim),nn.ReLU())
        self.classifier=nn.Linear(latent_dim,n_classes)
    def forward(self,x,return_aux=False):
        g=self.gate(x); z=self.encoder(x*g); logits=self.classifier(z)
        if return_aux:return logits,z,g
        return logits
    def expand_classes(self,new_n:int):
        if new_n<=self.n_classes:return
        old=self.classifier; new=nn.Linear(old.in_features,new_n).to(old.weight.device)
        with torch.no_grad():
            new.weight[:self.n_classes]=old.weight; new.bias[:self.n_classes]=old.bias
        self.classifier=new; self.n_classes=new_n
