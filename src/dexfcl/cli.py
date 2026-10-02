from __future__ import annotations
import argparse,copy
from pathlib import Path
import pandas as pd
from .reporting import aggregate_results,make_figures

def main():
    ap=argparse.ArgumentParser(prog='dexfcl'); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare'); p.add_argument('--dataset',choices=['ciciot2023','edgeiiotset'],required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--cap',type=int,default=120000); p.add_argument('--seed',type=int,default=2026)
    r=sp.add_parser('run'); r.add_argument('--config',required=True); r.add_argument('--protocol',choices=['closed','noniid','continual','zeroday','drift','explanation','fewshot','communication'],required=True); r.add_argument('--seed',type=int,required=True)
    a=sp.add_parser('aggregate'); a.add_argument('--results',default='results'); a.add_argument('--tables',default='tables'); a.add_argument('--figures',default='figures')
    a.add_argument('--source', choices=['auto','archived','campaign'], default='auto')
    p=sp.add_parser('reproduce-paper', help='Validate archived CSVs and regenerate tables/figures; does not train models')
    p.add_argument('--results', default='results'); p.add_argument('--tables', default='artifacts/paper/tables')
    p.add_argument('--figures', default='artifacts/paper/figures'); p.add_argument('--dpi', type=int, default=600)
    args=ap.parse_args()
    if args.cmd=='reproduce-paper':
        from .paper_outputs import reproduce_archive
        manifest=reproduce_archive(args.results,args.tables,args.figures,args.dpi)
        print('Archived numerical reproduction completed:', manifest['row_counts'])
        print('Training provenance remains unverified; see RELEASE_STATUS.md.'); return
    if args.cmd=='aggregate':
        outs=aggregate_results(args.results,args.tables,args.source)
        make_figures(args.tables,args.results,args.figures,args.source)
        print('Aggregated:',', '.join(outs)); return
    from .utils import load_yaml,ensure_dir
    from .data import prepare_ciciot2023,prepare_edgeiiotset
    from .experiment import run_closed,run_noniid,run_continual,run_zeroday,run_drift,run_explanation,run_fewshot,run_communication
    if args.cmd=='prepare':
        meta=prepare_ciciot2023(args.input,args.output,args.cap,seed=args.seed) if args.dataset=='ciciot2023' else prepare_edgeiiotset(args.input,args.output,args.cap,seed=args.seed); print(meta); return
    if args.cmd=='run':
        cfg=load_yaml(args.config); out=ensure_dir(cfg['paths']['results']); npz=cfg['paths']['processed']
        if args.protocol=='closed': df=run_closed(cfg,npz,out,args.seed); df.to_csv(out/f"closed_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='noniid': df=run_noniid(cfg,npz,out,args.seed); df.to_csv(out/f"noniid_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='continual':
            df,st=run_continual(cfg,npz,out,args.seed); df.to_csv(out/f"continual_{cfg['dataset']}_seed{args.seed}.csv",index=False); st.to_csv(out/f"continual_stages_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='zeroday':
            df=run_zeroday(cfg,npz,out,args.seed); df.to_csv(out/f"zeroday_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='drift':
            df,series=run_drift(cfg,npz,out,args.seed); df.to_csv(out/f"drift_{cfg['dataset']}_seed{args.seed}.csv",index=False); series.to_csv(out/f"drift_series_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='explanation':
            df=run_explanation(cfg,npz,out,args.seed); df.to_csv(out/f"explanation_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        elif args.protocol=='fewshot':
            df=run_fewshot(cfg,npz,out,args.seed); df.to_csv(out/f"fewshot_{cfg['dataset']}_seed{args.seed}.csv",index=False)
        else:
            df=run_communication(cfg,npz); df.to_csv(out/f"communication_{cfg['dataset']}.csv",index=False)
        print(df.to_string(index=False)); return
if __name__=='__main__':main()
