from __future__ import annotations
import numpy as np
import torch
from .model import DExNet
from .training import evaluate

def collect_outputs(model,X,y,idx,batch,device):
    model.eval(); logits=[]; zs=[]; ys=[]
    with torch.no_grad():
        for start in range(0,len(idx),batch):
            ii=idx[start:start+batch]; xb=torch.from_numpy(X[ii]).float().to(device)
            lg,z,_=model(xb,True); logits.append(lg.cpu()); zs.append(z.cpu()); ys.append(y[ii])
    return torch.cat(logits).numpy(),torch.cat(zs).numpy(),np.concatenate(ys)

def energy_score(logits,T=1.0):
    x=torch.from_numpy(logits)/T; return (-T*torch.logsumexp(x,dim=1)).numpy()

def msp_score(logits):
    p=torch.softmax(torch.from_numpy(logits),1).numpy(); return 1-p.max(1)

def prototype_score(z,prototypes):
    keys=sorted(prototypes); P=np.stack([prototypes[k].numpy() if hasattr(prototypes[k],'numpy') else np.asarray(prototypes[k]) for k in keys])
    # Euclidean distance after latent-space standardization by prototype dispersion.
    scale=np.std(P,axis=0)+1e-6; d=((z[:,None,:]-P[None,:,:])/scale[None,None,:])**2
    return np.sqrt(d.sum(2)).min(1)

def robust_normalize(x,reference):
    lo,hi=np.quantile(reference,[.01,.99]); return np.clip((x-lo)/(hi-lo+1e-12),0,1)
