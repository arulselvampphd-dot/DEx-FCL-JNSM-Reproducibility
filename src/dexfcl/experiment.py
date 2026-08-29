from __future__ import annotations
from pathlib import Path
import copy, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from .data import DatasetBundle,stratified_split_scale
from .partition import dirichlet_partition,save_partition_set
from .federated import run_federated
from .model import DExNet
from .training import evaluate,class_prototypes
from .metrics import multiclass_metrics,binary_open_metrics
from .open_set import collect_outputs,energy_score,msp_score,prototype_score,robust_normalize
from .mappings import CICIOT_CONTINUAL,EDGE_CONTINUAL
from .utils import seed_everything,ensure_dir,device_from_config,save_json,state_to_cpu

def _parts(y,train_idx,val_idx,K,alpha,seed):
    tr=dirichlet_partition(train_idx,y,K,alpha,seed,min_size=16)
    va=dirichlet_partition(val_idx,y,K,alpha,seed+1000,min_size=4)
    return tr,va

def run_closed(config,dataset_npz,out_dir,seed,methods=None,alpha=None):
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed)
    K=int(config['federated']['clients']); a=float(alpha if alpha is not None else config['federated']['alpha'])
    trp,vap=_parts(b.y,tr,va,K,a,seed); save_partition_set(config['paths']['partitions'],config['dataset'],seed,a,trp,vap,b.y,b.groups)
    device=device_from_config(config['runtime'].get('device','auto')); methods=methods or config['baselines']
    rows=[]
    for method in methods:
        t=time.time(); state,h,p,g,log,last_states,last_stats=run_federated(X,b.y,trp,vap,te,X.shape[1],len(b.groups),config['training'],method,device,seed)
        model=DExNet(X.shape[1],len(b.groups),config['training'].get('hidden1',128),config['training'].get('latent_dim',64),config['training'].get('dropout',.2)).to(device); model.load_state_dict(state)
        train_seconds=time.time()-t
        if device.type=='cuda': torch.cuda.synchronize()
        ti=time.perf_counter(); yt,yp,pr,_,_,_=evaluate(model,X,b.y,te,int(config['training']['batch_size']),device)
        if device.type=='cuda': torch.cuda.synchronize()
        infer_ms=1000*(time.perf_counter()-ti)/max(1,len(te)); m=multiclass_metrics(yt,yp,pr)
        m.update({"dataset":config['dataset'],"seed":seed,"alpha":a,"method":method,"train_seconds":train_seconds,"inference_ms_per_sample":infer_ms}); rows.append(m)
        pd.DataFrame(log).to_csv(Path(out_dir)/f"rounds_{config['dataset']}_{method}_seed{seed}_a{a:g}.csv",index=False)
    return pd.DataFrame(rows)

def run_noniid(config,dataset_npz,out_dir,seed):
    frames=[]
    for a in config['protocols']['noniid_alphas']:
        frames.append(run_closed(config,dataset_npz,out_dir,seed,methods=config['protocols']['noniid_methods'],alpha=float(a)))
    return pd.concat(frames,ignore_index=True)

def run_continual(config,dataset_npz,out_dir,seed):
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed)
    tasks=CICIOT_CONTINUAL if config['dataset']=='ciciot2023' else EDGE_CONTINUAL; name_to_id={g:i for i,g in enumerate(b.groups)}
    device=device_from_config(config['runtime'].get('device','auto')); K=int(config['federated']['clients']); a=float(config['federated']['alpha'])
    rows=[]; stages=[]
    for method in config['protocols']['continual_methods']:
        state=None; histories=None; best={}; task_perf=[]
        for t,task_names in enumerate(tasks):
            current_ids=[name_to_id[n] for n in task_names]; stage_classes=[name_to_id[n] for task in tasks[:t+1] for n in task]
            stage_train=tr[np.isin(b.y[tr],current_ids)]; stage_val=va[np.isin(b.y[va],current_ids)]
            trp,vap=_parts(b.y,stage_train,stage_val,K,a,seed+31*t)
            previous_ids=[name_to_id[n] for task in tasks[:t] for n in task]
            if previous_ids:
                hist_val=va[np.isin(b.y[va],previous_ids)]
                hvp=dirichlet_partition(hist_val,b.y,K,a,seed+5000+31*t,min_size=2)
            else:
                hvp=[np.array([],dtype=np.int64) for _ in range(K)]
            # Models retain full output dimension so global label IDs never change.
            state,histories,p,g,log,last_states,last_stats=run_federated(X,b.y,trp,vap,te[np.isin(b.y[te],stage_classes)],X.shape[1],len(b.groups),config['continual_training'],method,device,seed,rounds=config['continual_training']['rounds_per_task'],init_state=state,histories=histories,historical_val_parts=hvp)
            model=DExNet(X.shape[1],len(b.groups),config['continual_training'].get('hidden1',128),config['continual_training'].get('latent_dim',64),config['continual_training'].get('dropout',.2)).to(device); model.load_state_dict(state)
            per=[]
            for j,old_task in enumerate(tasks[:t+1]):
                ids=[name_to_id[n] for n in old_task]; idx=te[np.isin(b.y[te],ids)]
                yt,yp,pr,_,_,_=evaluate(model,X,b.y,idx,int(config['continual_training']['batch_size']),device); f=float(f1_score(yt,yp,average='macro',zero_division=0)); per.append(f); best[j]=max(best.get(j,0),f)
                stages.append({"dataset":config['dataset'],"seed":seed,"method":method,"stage":t+1,"eval_task":j+1,"macro_f1":f})
            task_perf=per
        forgetting=float(np.mean([best[j]-task_perf[j] for j in range(len(task_perf)-1)])) if len(task_perf)>1 else 0.
        bwt=float(np.mean([task_perf[j]-stages[[i for i,s in enumerate(stages) if s['method']==method and s['eval_task']==j+1 and s['stage']==j+1][0]]['macro_f1'] for j in range(len(task_perf)-1)])) if len(task_perf)>1 else 0.
        all_seen=[name_to_id[n] for task in tasks for n in task]; idx=te[np.isin(b.y[te],all_seen)]
        yt,yp,pr,_,_,_=evaluate(model,X,b.y,idx,int(config['continual_training']['batch_size']),device)
        rows.append({"dataset":config['dataset'],"seed":seed,"method":method,"final_macro_f1":f1_score(yt,yp,average='macro',zero_division=0),"forgetting":forgetting,"bwt":bwt})
    return pd.DataFrame(rows),pd.DataFrame(stages)

def run_zeroday(config,dataset_npz,out_dir,seed):
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed); device=device_from_config(config['runtime'].get('device','auto'))
    benign='Benign' if config['dataset']=='ciciot2023' else 'Normal'; holds=[g for g in b.groups if g!=benign]; rows=[]; raw_dir=ensure_dir(Path(out_dir)/'zeroday_scores')
    for hold in holds:
        hid=b.groups.index(hold); known_ids=[i for i in range(len(b.groups)) if i!=hid]
        # Remap known labels to compact 0..C-1 for training.
        remap={old:new for new,old in enumerate(known_ids)}; y2=np.full_like(b.y,-1); [y2.__setitem__(b.y==old,new) for old,new in remap.items()]
        known_tr=tr[b.y[tr]!=hid]; known_va=va[b.y[va]!=hid]; K=int(config['federated']['clients']); a=float(config['federated']['alpha'])
        trp=dirichlet_partition(known_tr,y2,K,a,seed,min_size=16); vap=dirichlet_partition(known_va,y2,K,a,seed+1000,min_size=4)
        cfg=copy.deepcopy(config['zeroday_training']); state,hist,p,g,log,last_states,last_stats=run_federated(X,y2,trp,vap,te[b.y[te]!=hid],X.shape[1],len(known_ids),cfg,'dexfcl',device,seed)
        model=DExNet(X.shape[1],len(known_ids),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); model.load_state_dict(state)
        # Recompute prototypes on all known train data to ensure coverage.
        protos=class_prototypes(model,X,y2,known_tr,int(cfg['batch_size']),device)
        lg_v,z_v,_=collect_outputs(model,X,y2,known_va,int(cfg['batch_size']),device)
        test_mix=np.concatenate([te[b.y[te]!=hid],te[b.y[te]==hid]])
        lg,z,_=collect_outputs(model,X,y2,test_mix,int(cfg['batch_size']),device); yu=(b.y[test_mix]==hid).astype(int)
        scores_v={"msp":msp_score(lg_v),"energy":energy_score(lg_v),"prototype":prototype_score(z_v,protos)}
        scores={"msp":msp_score(lg),"energy":energy_score(lg),"prototype":prototype_score(z,protos)}
        # energy direction: more positive/less negative energy is more OOD; robust normalization handles ordering, but ensure correlation with MSP direction by validation quantiles only.
        nv={k:robust_normalize(v,scores_v[k]) for k,v in scores.items()}; vv={k:robust_normalize(scores_v[k],scores_v[k]) for k in scores_v}
        beta=float(config['protocols'].get('open_beta',.5)); nv['combined']=beta*nv['energy']+(1-beta)*nv['prototype']; vv['combined']=beta*vv['energy']+(1-beta)*vv['prototype']
        for name in ['msp','energy','prototype','combined']:
            th=float(np.quantile(vv[name],config['protocols'].get('open_quantile',.95))); m=binary_open_metrics(yu,nv[name],th); m.update({"dataset":config['dataset'],"seed":seed,"held_out":hold,"detector":name,"threshold":th}); rows.append(m)
        np.savez_compressed(raw_dir/f"{config['dataset']}_seed{seed}_{hold}.npz",unknown=yu,val_msp=vv['msp'],val_energy=vv['energy'],val_prototype=vv['prototype'],**nv)
    return pd.DataFrame(rows)

def _window_stats(model,X,idx,window,device,batch):
    vals=[]
    for s in range(0,len(idx)-window+1,window):
        w=idx[s:s+window]
        _,_,_,ent,lat,_=evaluate(model,X,np.zeros(len(X),dtype=np.int64),w,batch,device)
        vals.append((ent,lat.numpy()))
    return vals

def run_drift(config,dataset_npz,out_dir,seed):
    """Real-record drift test: a held-out attack family is introduced into the post-change stream without modifying features."""
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed)
    family=config['protocols']['drift_family']; hid=b.groups.index(family); known=[i for i in range(len(b.groups)) if i!=hid]
    remap={old:new for new,old in enumerate(known)}; y2=np.full_like(b.y,-1)
    for old,new in remap.items(): y2[b.y==old]=new
    known_tr=tr[b.y[tr]!=hid]; known_va=va[b.y[va]!=hid]; known_te=te[b.y[te]!=hid]; unk_te=te[b.y[te]==hid]
    K=int(config['federated']['clients']); a=float(config['federated']['alpha']); trp=dirichlet_partition(known_tr,y2,K,a,seed,min_size=16); vap=dirichlet_partition(known_va,y2,K,a,seed+1000,min_size=4)
    device=device_from_config(config['runtime'].get('device','auto')); cfg=copy.deepcopy(config['zeroday_training'])
    state,hist,p,g,log,last_states,last_stats=run_federated(X,y2,trp,vap,known_te,X.shape[1],len(known),cfg,'dexfcl',device,seed)
    model=DExNet(X.shape[1],len(known),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); model.load_state_dict(state)
    rng=np.random.default_rng(seed+5000); window=int(config['protocols'].get('drift_window',256)); nwin=8
    npre=min(len(known_te),window*nwin); pre=rng.choice(known_te,npre,replace=False)
    # Post-change contains 50% unseen-family traffic and 50% known traffic; no feature perturbation is applied.
    half=window*nwin//2; u=rng.choice(unk_te,min(len(unk_te),half),replace=len(unk_te)<half); k=rng.choice(known_te,window*nwin-len(u),replace=len(known_te)<window*nwin-len(u)); post=np.concatenate([u,k]); rng.shuffle(post)
    stream=np.concatenate([pre,post]); stats=_window_stats(model,X,stream,window,device,int(cfg['batch_size']))
    H=np.array([v[0] for v in stats]); Z=[v[1] for v in stats]; calib=max(3,len(pre)//window//2)
    href=H[:calib].mean(); zref=np.mean(np.stack(Z[:calib]),axis=0)
    dh=np.abs(H-href)/(abs(href)+1e-8); dz=np.array([1-np.dot(z,zref)/(np.linalg.norm(z)*np.linalg.norm(zref)+1e-8) for z in Z])
    pre_eval=slice(0,len(pre)//window); th_h=dh[:calib].mean()+3*dh[:calib].std()+1e-8; th_z=dz[:calib].mean()+3*dz[:calib].std()+1e-8
    eta=float(cfg.get('drift_eta',.5)); dual=eta*(dh/th_h)+(1-eta)*(dz/th_z); signals={'entropy':dh/th_h,'latent':dz/th_z,'dual':dual}
    rows=[]; change_win=len(pre)//window
    series=[]
    for name,sig in signals.items():
        alarms=np.flatnonzero(sig>1.0); post_alarms=alarms[alarms>=change_win]; detected=len(post_alarms)>0; first=int(post_alarms[0]) if detected else -1
        rows.append({'dataset':config['dataset'],'seed':seed,'drift_family':family,'detector':name,'detected':int(detected),
                     'false_alarm_rate':float(np.mean(sig[:change_win]>1.0)),'delay_samples':int((first-change_win)*window if detected else len(post))})
        for i,v in enumerate(sig): series.append({'dataset':config['dataset'],'seed':seed,'detector':name,'window':i,'score':float(v),'change_window':change_win})
    return pd.DataFrame(rows),pd.DataFrame(series)

def _pairwise_explanation_metrics(vectors_by_client,topk=(5,10,15)):
    from scipy.stats import spearmanr,kendalltau
    vals={'spearman':[],'kendall':[]};
    for k in topk: vals[f'jaccard{k}']=[]
    clients=sorted(vectors_by_client)
    for i in range(len(clients)):
        for j in range(i+1,len(clients)):
            a=vectors_by_client[clients[i]]; b=vectors_by_client[clients[j]]
            vals['spearman'].append(float(spearmanr(a,b).statistic)); vals['kendall'].append(float(kendalltau(a,b).statistic))
            for k in topk:
                kk=min(k,len(a)); A=set(np.argsort(np.abs(a))[-kk:]); B=set(np.argsort(np.abs(b))[-kk:]); vals[f'jaccard{k}'].append(len(A&B)/max(1,len(A|B)))
    return {k:float(np.nanmean(v)) for k,v in vals.items()}

def run_explanation(config,dataset_npz,out_dir,seed):
    from .training import class_gate_means
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed); device=device_from_config(config['runtime'].get('device','auto'))
    a=float(config['protocols'].get('explanation_alpha',0.1)); K=int(config['federated']['clients']); trp,vap=_parts(b.y,tr,va,K,a,seed); rows=[]
    for method in ['fedavg','dexfcl']:
        cfg=copy.deepcopy(config['training']); cfg['lambda_exp']=float(config['protocols'].get('explanation_lambda',0.10)) if method=='dexfcl' else 0.0
        state,h,p,g,log,last_states,last_stats=run_federated(X,b.y,trp,vap,te,X.shape[1],len(b.groups),cfg,method,device,seed)
        vectors={}
        for k,s in enumerate(last_states):
            m=DExNet(X.shape[1],len(b.groups),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); m.load_state_dict(s)
            means=class_gate_means(m,X,b.y,vap[k],int(cfg['batch_size']),device)
            if means: vectors[k]=np.mean(np.stack([v.numpy() for v in means.values()]),axis=0)
        met=_pairwise_explanation_metrics(vectors)
        model=DExNet(X.shape[1],len(b.groups),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); model.load_state_dict(state)
        yt,yp,pr,_,_,_=evaluate(model,X,b.y,te,int(cfg['batch_size']),device); met.update({'dataset':config['dataset'],'seed':seed,'method':method,'macro_f1':f1_score(yt,yp,average='macro',zero_division=0)}); rows.append(met)
    return pd.DataFrame(rows)

def run_fewshot(config,dataset_npz,out_dir,seed):
    """Federated unknown-to-known assimilation using held-out training records as verified support; final test remains untouched."""
    seed_everything(seed); b=DatasetBundle.load(dataset_npz); X,tr,va,te,_=stratified_split_scale(b,seed); family=config['protocols']['fewshot_family']; hid=b.groups.index(family)
    known=[i for i in range(len(b.groups)) if i!=hid]; remap={old:new for new,old in enumerate(known)}; y_known=np.full_like(b.y,-1)
    for old,new in remap.items(): y_known[b.y==old]=new
    known_tr=tr[b.y[tr]!=hid]; known_va=va[b.y[va]!=hid]; K=int(config['federated']['clients']); a=float(config['federated']['alpha']); trp=dirichlet_partition(known_tr,y_known,K,a,seed,min_size=16); vap=dirichlet_partition(known_va,y_known,K,a,seed+1000,min_size=4)
    device=device_from_config(config['runtime'].get('device','auto')); cfg=copy.deepcopy(config['zeroday_training'])
    state,hist,p,g,log,last_states,last_stats=run_federated(X,y_known,trp,vap,te[b.y[te]!=hid],X.shape[1],len(known),cfg,'dexfcl',device,seed)
    # Expand classifier; known compact class IDs remain unchanged and the new family becomes the last class.
    base=DExNet(X.shape[1],len(known),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); base.load_state_dict(state); base.expand_classes(len(known)+1); expanded=state_to_cpu(base.state_dict())
    # Histories may contain old classifier tensors. local_train skips incompatible shapes.
    y3=y_known.copy(); y3[b.y==hid]=len(known); support_pool=tr[b.y[tr]==hid]; rng=np.random.default_rng(seed+7000); rng.shuffle(support_pool)
    rows=[]; replay=int(config['protocols'].get('fewshot_replay_per_client',2000)); rounds=int(config['protocols'].get('fewshot_rounds',5))
    for shots in config['protocols']['fewshot_counts']:
        chosen=support_pool[:min(int(shots),len(support_pool))]; pieces=np.array_split(chosen,K); aug=[]
        for k in range(K):
            base_idx=trp[k]; keep=base_idx if len(base_idx)<=replay else rng.choice(base_idx,replay,replace=False); aug.append(np.concatenate([keep,pieces[k]]).astype(np.int64))
        st,hh,pp,gg,ll,ls,lstat=run_federated(X,y3,aug,vap,te,X.shape[1],len(known)+1,cfg,'dexfcl',device,seed,rounds=rounds,init_state=expanded,histories=copy.deepcopy(hist),historical_val_parts=vap)
        m=DExNet(X.shape[1],len(known)+1,cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2)).to(device); m.load_state_dict(st)
        new_te=te[b.y[te]==hid]; yt,yp,pr,_,_,_=evaluate(m,X,y3,new_te,int(cfg['batch_size']),device); new_f1=float(f1_score(yt,yp,average='macro',zero_division=0))
        hist_te=te[b.y[te]!=hid]; yt2,yp2,pr2,_,_,_=evaluate(m,X,y3,hist_te,int(cfg['batch_size']),device); hist_f1=float(f1_score(yt2,yp2,average='macro',zero_division=0))
        rows.append({'dataset':config['dataset'],'seed':seed,'family':family,'shots':int(shots),'new_class_f1':new_f1,'historical_macro_f1':hist_f1})
    return pd.DataFrame(rows)

def run_communication(config,dataset_npz):
    b=DatasetBundle.load(dataset_npz); cfg=config['training']; model=DExNet(b.X.shape[1],len(b.groups),cfg.get('hidden1',128),cfg.get('latent_dim',64),cfg.get('dropout',.2))
    P=sum(p.numel() for p in model.parameters()); bpe=4; model_bytes=P*bpe; proto_bytes=len(b.groups)*cfg.get('latent_dim',64)*bpe; attr_bytes=len(b.groups)*b.X.shape[1]*bpe; meta_bytes=4*bpe
    return pd.DataFrame([{'dataset':config['dataset'],'parameters':P,'model_bytes_per_client_round':model_bytes,'prototype_bytes':proto_bytes,'attribution_bytes':attr_bytes,'metadata_bytes':meta_bytes,'dexfcl_bytes_per_client_round':model_bytes+proto_bytes+attr_bytes+meta_bytes,'overhead_pct':100*(proto_bytes+attr_bytes+meta_bytes)/model_bytes}])
