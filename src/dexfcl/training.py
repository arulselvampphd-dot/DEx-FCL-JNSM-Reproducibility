from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import copy, math
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score
from .utils import state_to_cpu

@dataclass
class ClientHistory:
    fisher: dict[str,torch.Tensor]|None=None
    old_params: dict[str,torch.Tensor]|None=None
    prototypes: dict[int,torch.Tensor]|None=None
    explanation_anchors: dict[int,torch.Tensor]|None=None
    reference_entropy: float|None=None
    reference_latent: torch.Tensor|None=None
    best_hist_f1: float=0.0
    scaffold_c: dict[str,torch.Tensor]|None=None

def loader(X,y,idx,batch,shuffle,device=None):
    ds=TensorDataset(torch.from_numpy(X[idx]).float(),torch.from_numpy(y[idx]).long())
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)

def evaluate(model,X,y,idx,batch,device,known_classes=None):
    model.eval(); ys=[]; ps=[]; probs=[]; ents=[]; zs=[]; gates=[]
    with torch.no_grad():
        for xb,yb in loader(X,y,idx,batch,False):
            xb=xb.to(device); logits,z,g=model(xb,True)
            prob=torch.softmax(logits,1); pred=prob.argmax(1)
            ys.append(yb.numpy()); ps.append(pred.cpu().numpy()); probs.append(prob.cpu().numpy())
            ents.append((-(prob*torch.log(prob.clamp_min(1e-12))).sum(1)).cpu().numpy())
            zs.append(z.cpu()); gates.append(g.cpu())
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(probs),float(np.concatenate(ents).mean()),torch.cat(zs).mean(0),torch.cat(gates)

def class_prototypes(model,X,y,idx,batch,device):
    model.eval(); sums={}; counts=defaultdict(int)
    with torch.no_grad():
        for xb,yb in loader(X,y,idx,batch,False):
            xb=xb.to(device); _,z,_=model(xb,True); z=z.cpu()
            for c in yb.unique().tolist():
                m=(yb==c); s=z[m].sum(0); sums[c]=s if c not in sums else sums[c]+s; counts[c]+=int(m.sum())
    return {int(c):sums[c]/counts[c] for c in sums}

def class_gate_means(model,X,y,idx,batch,device):
    model.eval(); sums={}; counts=defaultdict(int)
    with torch.no_grad():
        for xb,yb in loader(X,y,idx,batch,False):
            xb=xb.to(device); _,_,g=model(xb,True); g=g.cpu()
            for c in yb.unique().tolist():
                m=(yb==c); s=g[m].sum(0); sums[c]=s if c not in sums else sums[c]+s; counts[c]+=int(m.sum())
    return {int(c):sums[c]/counts[c] for c in sums}

def estimate_fisher(model,X,y,idx,batch,device,max_batches=8):
    model.eval(); fisher={n:torch.zeros_like(p,device='cpu') for n,p in model.named_parameters() if p.requires_grad}; n_b=0
    for xb,yb in loader(X,y,idx,batch,True):
        xb=xb.to(device); yb=yb.to(device); model.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(xb),yb); loss.backward(); n_b+=1
        for n,p in model.named_parameters():
            if n in fisher and p.grad is not None: fisher[n]+=p.grad.detach().cpu().pow(2)
        if n_b>=max_batches:break
    for n in fisher:fisher[n]/=max(1,n_b)
    return fisher

def cosine_loss(a,b):
    return 1-torch.nn.functional.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0),dim=1).mean()

def local_train(model,global_state,X,y,train_idx,val_idx,history:ClientHistory,cfg,device,method='fedavg',global_scaffold=None,hist_val_idx=None):
    model.load_state_dict(global_state); model.to(device); model.train()
    opt_name=('sgd' if method=='scaffold' else cfg.get('optimizer','adamw').lower()); lr=float(cfg['lr'])
    opt=(torch.optim.SGD(model.parameters(),lr=lr,momentum=cfg.get('momentum',0.0)) if opt_name=='sgd'
         else torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg.get('weight_decay',1e-4)))
    old_global={n:p.detach().clone() for n,p in model.named_parameters()}
    pre_hist_quality=None
    if hist_val_idx is not None and len(hist_val_idx)>0:
        hy,hp,_,_,_,_=evaluate(model,X,y,hist_val_idx,int(cfg['batch_size']),device)
        pre_hist_quality=float(f1_score(hy,hp,average='macro',zero_division=0))
    steps=0
    ce_fn=torch.nn.CrossEntropyLoss()
    for _ in range(int(cfg['local_epochs'])):
        for xb,yb in loader(X,y,train_idx,int(cfg['batch_size']),True):
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); logits,z,g=model(xb,True); loss=ce_fn(logits,yb)
            if method=='fedprox':
                mu=float(cfg.get('fedprox_mu',0.01)); prox=0.
                for n,p in model.named_parameters(): prox=prox+(p-old_global[n]).pow(2).sum()
                loss=loss+.5*mu*prox
            if method in {'ewc','dexfcl'} and history.fisher and history.old_params:
                ewc=0.
                for n,p in model.named_parameters():
                    if n in history.fisher and n in history.old_params and history.old_params[n].shape==p.shape and history.fisher[n].shape==p.shape:
                        ewc=ewc+(history.fisher[n].to(device)*(p-history.old_params[n].to(device)).pow(2)).sum()
                loss=loss+float(cfg.get('lambda_ewc',0.0))*ewc
            if method=='dexfcl' and history.prototypes:
                ploss=0.; pc=0
                for c in yb.unique().tolist():
                    if int(c) in history.prototypes:
                        cur=z[yb==c].mean(0); old=history.prototypes[int(c)].to(device)
                        ploss=ploss+torch.nn.functional.mse_loss(cur,old); pc+=1
                if pc: loss=loss+float(cfg.get('lambda_proto',0.0))*ploss/pc
            if method=='dexfcl' and history.explanation_anchors:
                eloss=0.; ec=0
                for c in yb.unique().tolist():
                    if int(c) in history.explanation_anchors:
                        cur=g[yb==c].mean(0); anc=history.explanation_anchors[int(c)].to(device)
                        eloss=eloss+cosine_loss(cur,anc); ec+=1
                if ec: loss=loss+float(cfg.get('lambda_exp',0.0))*eloss/ec
            loss.backward()
            # SCAFFOLD control-variate correction. Use SGD for canonical behavior.
            if method=='scaffold' and global_scaffold is not None and history.scaffold_c is not None:
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(global_scaffold[n].to(device)-history.scaffold_c[n].to(device))
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get('grad_clip',5.0))); opt.step(); steps+=1
    local_state=state_to_cpu(model.state_dict())
    # local quality on client validation
    yt,yp,pr,entropy,latent,_=evaluate(model,X,y,val_idx,int(cfg['batch_size']),device)
    quality=float(f1_score(yt,yp,average='macro',zero_division=0))
    if hist_val_idx is not None and len(hist_val_idx)>0 and pre_hist_quality is not None:
        hy2,hp2,_,_,_,_=evaluate(model,X,y,hist_val_idx,int(cfg['batch_size']),device)
        post_hist_quality=float(f1_score(hy2,hp2,average='macro',zero_division=0))
        forgetting=max(0.0,pre_hist_quality-post_hist_quality)
    else:
        forgetting=0.0
    drift=0.0
    if history.reference_entropy is not None and history.reference_latent is not None:
        dh=abs(entropy-history.reference_entropy)/(abs(history.reference_entropy)+1e-8)
        a=latent; b=history.reference_latent; dz=float(1-torch.dot(a,b)/(a.norm()*b.norm()+1e-8))
        drift=float(cfg.get('drift_eta',0.5))*dh+(1-float(cfg.get('drift_eta',0.5)))*max(0,dz)
    protos=class_prototypes(model,X,y,train_idx,int(cfg['batch_size']),device)
    gates=class_gate_means(model,X,y,train_idx,int(cfg['batch_size']),device)
    scaffold_delta=None
    if method=='scaffold' and global_scaffold is not None and history.scaffold_c is not None:
        c_new={}; scaffold_delta={}
        denom=max(1,steps*lr)
        for n,p in model.named_parameters():
            c_new[n]=history.scaffold_c[n]-global_scaffold[n].cpu()+(old_global[n].detach().cpu()-p.detach().cpu())/denom
            scaffold_delta[n]=c_new[n]-history.scaffold_c[n]
        history.scaffold_c=c_new
    history.reference_entropy=entropy; history.reference_latent=latent.detach().cpu(); history.best_hist_f1=max(history.best_hist_f1,quality)
    return local_state,{"quality":quality,"forgetting":forgetting,"drift":drift,"n":len(train_idx),"prototypes":protos,"gates":gates,"scaffold_delta":scaffold_delta},history
