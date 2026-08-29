from __future__ import annotations
import copy
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import f1_score
from .model import DExNet
from .training import ClientHistory,local_train,evaluate,estimate_fisher
from .utils import state_to_cpu

def weighted_average(states,weights):
    out={}; weights=np.asarray(weights,dtype=float); weights=weights/weights.sum()
    for k in states[0]:
        if torch.is_floating_point(states[0][k]): out[k]=sum(float(w)*s[k] for w,s in zip(weights,states))
        else: out[k]=states[0][k].clone()
    return out

def aggregate_class_vectors(stats,key,weights):
    sums={}; den=defaultdict(float)
    for st,w in zip(stats,weights):
        for c,v in st[key].items():
            if c not in sums:sums[c]=v*float(w)
            else:sums[c]+=v*float(w)
            den[c]+=float(w)
    return {c:sums[c]/max(den[c],1e-12) for c in sums}

def run_federated(X,y,train_parts,val_parts,test_idx,input_dim,n_classes,cfg,method,device,seed,rounds=None,init_state=None,histories=None,historical_val_parts=None):
    model=DExNet(input_dim,n_classes,cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device)
    if init_state is not None:model.load_state_dict(init_state)
    global_state=state_to_cpu(model.state_dict()); K=len(train_parts)
    if histories is None: histories=[ClientHistory() for _ in range(K)]
    if method=='scaffold':
        global_c={n:torch.zeros_like(p.detach().cpu()) for n,p in model.named_parameters()}
        for h in histories:
            if h.scaffold_c is None:h.scaffold_c={n:torch.zeros_like(v) for n,v in global_c.items()}
    else: global_c=None
    log=[]; global_protos={}; global_gates={}
    R=int(rounds or cfg['rounds'])
    for rnd in range(R):
        states=[]; stats=[]
        for k in range(K):
            m=DExNet(input_dim,n_classes,cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device)
            histories[k].prototypes=global_protos or histories[k].prototypes
            histories[k].explanation_anchors=global_gates or histories[k].explanation_anchors
            s,st,h=local_train(m,global_state,X,y,train_parts[k],val_parts[k],histories[k],cfg,device,method,global_c, None if historical_val_parts is None else historical_val_parts[k])
            histories[k]=h; states.append(s); stats.append(st)
        if method=='dexfcl' and cfg.get('adaptive_aggregation',True):
            raw=[]
            for st in stats:
                q=max(1e-6,st['quality']); score=st['n']*q*(1+float(cfg.get('gamma',.5))*st['drift'])/(1+float(cfg.get('rho',1.0))*st['forgetting'])
                raw.append(score)
            weights=np.asarray(raw)/max(np.sum(raw),1e-12)
        else:
            weights=np.asarray([st['n'] for st in stats],dtype=float); weights/=weights.sum()
        global_state=weighted_average(states,weights); model.load_state_dict(global_state)
        if method=='dexfcl':
            global_protos=aggregate_class_vectors(stats,'prototypes',weights); global_gates=aggregate_class_vectors(stats,'gates',weights)
        if method=='scaffold':
            deltas=[st['scaffold_delta'] for st in stats]
            for n in global_c:
                global_c[n]=global_c[n]+sum(d[n] for d in deltas if d is not None)/K
        yt,yp,pr,_,_,_=evaluate(model,X,y,test_idx,int(cfg['batch_size']),device)
        mf=float(f1_score(yt,yp,average='macro',zero_division=0)); log.append({"round":rnd+1,"macro_f1":mf})
    # update EWC reference after a stage for EWC/DEx-FCL
    if method in {'ewc','dexfcl'}:
        for k in range(K):
            m=DExNet(input_dim,n_classes,cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); m.load_state_dict(global_state)
            histories[k].fisher=estimate_fisher(m,X,y,train_parts[k],int(cfg['batch_size']),device,max_batches=int(cfg.get('fisher_batches',8)))
            histories[k].old_params={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
    return global_state,histories,global_protos,global_gates,log,states,stats
