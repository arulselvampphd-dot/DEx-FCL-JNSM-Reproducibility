from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

SEEDS=[7,19,31,43,59]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--protocols',nargs='+',default=['closed','noniid','continual','zeroday','drift','explanation','fewshot'])
    ap.add_argument('--seeds',nargs='*',type=int,default=SEEDS)
    ap.add_argument('--aggregate',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy(); env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    for protocol in args.protocols:
        for seed in args.seeds:
            cmd=[sys.executable,'-m','dexfcl','run','--config',args.config,'--protocol',protocol,'--seed',str(seed)]
            print('\n>>>',' '.join(cmd),flush=True)
            subprocess.run(cmd,cwd=root,env=env,check=True)
    if args.aggregate:
        subprocess.run([sys.executable,'-m','dexfcl','aggregate','--results','results','--tables','tables','--figures','figures'],cwd=root,env=env,check=True)
if __name__=='__main__':main()
