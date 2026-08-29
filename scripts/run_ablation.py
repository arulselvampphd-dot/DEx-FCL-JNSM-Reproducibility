from __future__ import annotations
import argparse, copy, os
from pathlib import Path
import pandas as pd
from dexfcl.utils import load_yaml,ensure_dir
from dexfcl.experiment import run_continual

SEEDS=[7,19,31,43,59]
VARIANTS={
 'Base FL': dict(lambda_ewc=0.0,lambda_proto=0.0,lambda_exp=0.0,adaptive_aggregation=False),
 '+ EWC': dict(lambda_ewc=0.05,lambda_proto=0.0,lambda_exp=0.0,adaptive_aggregation=False),
 '+ Prototype': dict(lambda_ewc=0.05,lambda_proto=0.20,lambda_exp=0.0,adaptive_aggregation=False),
 '+ Adaptive aggregation': dict(lambda_ewc=0.05,lambda_proto=0.20,lambda_exp=0.0,adaptive_aggregation=True),
 'Full DEx-FCL': dict(lambda_ewc=0.05,lambda_proto=0.20,lambda_exp=0.10,adaptive_aggregation=True),
}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seeds',nargs='*',type=int,default=SEEDS); args=ap.parse_args()
 base=load_yaml(args.config); out=ensure_dir(base['paths']['results']); rows=[]
 for seed in args.seeds:
  for name,params in VARIANTS.items():
   cfg=copy.deepcopy(base); cfg['protocols']['continual_methods']=['dexfcl']; cfg['continual_training'].update(params)
   df,_=run_continual(cfg,cfg['paths']['processed'],out,seed); r=df.iloc[0].to_dict(); r['variant']=name; rows.append(r)
   pd.DataFrame(rows).to_csv(out/f"ablation_{base['dataset']}.csv",index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
