from __future__ import annotations
import argparse,copy
from pathlib import Path
import pandas as pd
import numpy as np
from dexfcl.utils import load_yaml,ensure_dir
from dexfcl.experiment import run_continual,run_explanation
from dexfcl.metrics import binary_open_metrics

SEEDS=[7,19,31,43,59]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seeds',nargs='*',type=int,default=SEEDS); args=ap.parse_args()
 base=load_yaml(args.config); out=ensure_dir(base['paths']['results']); rows=[]
 for seed in args.seeds:
  for val in [0.0,0.01,0.05,0.10,0.20]:
   cfg=copy.deepcopy(base); cfg['protocols']['continual_methods']=['dexfcl']; cfg['continual_training']['lambda_ewc']=val
   df,_=run_continual(cfg,cfg['paths']['processed'],out,seed); r=df.iloc[0].to_dict(); r.update({'parameter':'lambda_ewc','value':val,'metric':'forgetting','score':r['forgetting']}); rows.append(r)
  for val in [0.0,0.05,0.10,0.20]:
   cfg=copy.deepcopy(base); cfg['protocols']['explanation_lambda']=val
   df=run_explanation(cfg,cfg['paths']['processed'],out,seed); r=df[df.method=='dexfcl'].iloc[0].to_dict(); r.update({'parameter':'lambda_exp','value':val,'metric':'kendall','score':r['kendall']}); rows.append(r)
  # Open-set mixing sensitivity reuses raw energy/prototype scores from the completed zero-day run.
  score_dir=Path(out)/'zeroday_scores'
  for file in score_dir.glob(f"{base['dataset']}_seed{seed}_*.npz"):
   z=np.load(file); hold=file.stem.split(f"seed{seed}_",1)[1]
   for beta in [0.25,0.50,0.75]:
    test=beta*z['energy']+(1-beta)*z['prototype']; val=beta*z['val_energy']+(1-beta)*z['val_prototype']; th=float(np.quantile(val,base['protocols'].get('open_quantile',.95)))
    m=binary_open_metrics(z['unknown'],test,th); rows.append({'dataset':base['dataset'],'seed':seed,'held_out':hold,'parameter':'open_beta','value':beta,'metric':'auroc','score':m['auroc']})
  pd.DataFrame(rows).to_csv(out/f"sensitivity_{base['dataset']}.csv",index=False)
 print(pd.DataFrame(rows)[['dataset','seed','parameter','value','metric','score']].to_string(index=False))
if __name__=='__main__':main()
