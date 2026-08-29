from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare,wilcoxon,rankdata

METHOD_LABELS={'fedavg':'FedAvg','fedprox':'FedProx','scaffold':'SCAFFOLD','ewc':'FedAvg+EWC','dexfcl':'DEx-FCL'}

def _holm_adjust(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*p[idx]); running=max(running,val); adj[idx]=running
    return adj

def _rank_biserial_paired(a,b):
    d=np.asarray(a)-np.asarray(b); d=d[d!=0]
    if len(d)==0:return 0.0
    r=rankdata(np.abs(d)); wp=r[d>0].sum(); wm=r[d<0].sum(); return float((wp-wm)/(wp+wm))

def aggregate_results(results_dir, tables_dir):
    r=Path(results_dir); t=Path(tables_dir); t.mkdir(parents=True,exist_ok=True)
    outs={}
    for kind in ['closed','noniid','continual','continual_stages','zeroday','drift','drift_series','explanation','fewshot','communication','ablation','sensitivity']:
        files=list(r.glob(f"{kind}_*.csv"))
        if files:
            df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True); df.to_csv(t/f"all_{kind}.csv",index=False); outs[kind]=df
    if 'closed' in outs:
        df=outs['closed']; num=['accuracy','precision','recall','macro_f1','mcc','auroc','train_seconds','inference_ms_per_sample']
        s=df.groupby(['dataset','method'])[num].agg(['mean','std']).reset_index(); s.to_csv(t/'table_overall.csv',index=False)
        # paired nonparametric tests per dataset on macro-F1
        stat=[]
        for ds,d in df.groupby('dataset'):
            piv=d.pivot(index='seed',columns='method',values='macro_f1').dropna()
            cols=[m for m in ['fedavg','fedprox','scaffold','ewc','dexfcl'] if m in piv.columns]
            if len(cols)>=3 and len(piv)>=3:
                fr=friedmanchisquare(*[piv[c] for c in cols]); stat.append({'dataset':ds,'comparison':'overall Friedman','statistic':fr.statistic,'p_value':fr.pvalue})
            if 'dexfcl' in piv:
                pair=[]
                for m in cols:
                    if m=='dexfcl':continue
                    try:
                        w=wilcoxon(piv['dexfcl'],piv[m]); pair.append({'dataset':ds,'comparison':f'DEx-FCL vs {METHOD_LABELS.get(m,m)}','statistic':w.statistic,'p_value':w.pvalue,'effect_rank_biserial':_rank_biserial_paired(piv['dexfcl'],piv[m])})
                    except Exception: pass
                if pair:
                    adj=_holm_adjust([x['p_value'] for x in pair])
                    for x,a in zip(pair,adj): x['p_holm']=a; x['significant_0.05']=bool(a<0.05); stat.append(x)
        pd.DataFrame(stat).to_csv(t/'table_statistics.csv',index=False)

    if 'noniid' in outs:
        outs['noniid'].groupby(['dataset','method','alpha'])['macro_f1'].agg(['mean','std']).reset_index().to_csv(t/'table_noniid.csv',index=False)
    if 'zeroday' in outs:
        outs['zeroday'].groupby(['dataset','detector'])[['auroc','aupr','unknown_recall','fpr','fpr95']].agg(['mean','std']).reset_index().to_csv(t/'table_zeroday_summary.csv',index=False)
        outs['zeroday'][outs['zeroday'].detector=='combined'].pivot_table(index=['dataset','held_out'],values=['auroc','aupr','unknown_recall','fpr95'],aggfunc=['mean','std']).to_csv(t/'table_zeroday_attackwise.csv')
    if 'continual' in outs:
        outs['continual'].groupby(['dataset','method'])[['final_macro_f1','forgetting','bwt']].agg(['mean','std']).reset_index().to_csv(t/'table_continual.csv',index=False)

    if 'drift' in outs:
        outs['drift'].groupby(['dataset','detector'])[['detected','false_alarm_rate','delay_samples']].agg(['mean','std']).reset_index().to_csv(t/'table_drift.csv',index=False)
    if 'explanation' in outs:
        outs['explanation'].groupby(['dataset','method'])[['spearman','kendall','jaccard5','jaccard10','jaccard15','macro_f1']].agg(['mean','std']).reset_index().to_csv(t/'table_explanation.csv',index=False)
    if 'fewshot' in outs:
        outs['fewshot'].groupby(['dataset','family','shots'])[['new_class_f1','historical_macro_f1']].agg(['mean','std']).reset_index().to_csv(t/'table_fewshot.csv',index=False)
    if 'communication' in outs:
        outs['communication'].drop_duplicates().to_csv(t/'table_communication.csv',index=False)

    if 'ablation' in outs:
        outs['ablation'].groupby(['dataset','variant'])[['final_macro_f1','forgetting','bwt']].agg(['mean','std']).reset_index().to_csv(t/'table_ablation.csv',index=False)
    if 'sensitivity' in outs:
        outs['sensitivity'].groupby(['dataset','parameter','value','metric'])['score'].agg(['mean','std']).reset_index().to_csv(t/'table_sensitivity.csv',index=False)
    return outs

def make_figures(tables_dir,results_dir,figures_dir):
    t=Path(tables_dir); r=Path(results_dir); f=Path(figures_dir); f.mkdir(parents=True,exist_ok=True)
    if (t/'all_closed.csv').exists():
        df=pd.read_csv(t/'all_closed.csv')
        for ds,d in df.groupby('dataset'):
            s=d.groupby('method').macro_f1.agg(['mean','std']).sort_values('mean',ascending=False)
            fig,ax=plt.subplots(figsize=(6.8,4.2)); ax.bar([METHOD_LABELS.get(i,i) for i in s.index],s['mean'],yerr=s['std'])
            ax.set_ylabel('Macro-F1'); ax.set_ylim(0,1); ax.set_title(f'Federated IDS performance — {ds}'); ax.tick_params(axis='x',rotation=20)
            fig.tight_layout(); fig.savefig(f/f'fig_overall_{ds}.png',dpi=300); fig.savefig(f/f'fig_overall_{ds}.pdf'); plt.close(fig)
    if (t/'all_noniid.csv').exists():
        df=pd.read_csv(t/'all_noniid.csv')
        for ds,d in df.groupby('dataset'):
            fig,ax=plt.subplots(figsize=(6.3,4.1))
            for m,g in d.groupby('method'):
                s=g.groupby('alpha').macro_f1.agg(['mean','std']).sort_index(); ax.errorbar(s.index,s['mean'],yerr=s['std'],marker='o',label=METHOD_LABELS.get(m,m))
            ax.set_xlabel('Dirichlet α'); ax.set_ylabel('Macro-F1'); ax.set_title(f'Non-IID sensitivity — {ds}'); ax.legend(); fig.tight_layout(); fig.savefig(f/f'fig_noniid_{ds}.png',dpi=300); fig.savefig(f/f'fig_noniid_{ds}.pdf'); plt.close(fig)
    if (t/'all_continual.csv').exists():
        df=pd.read_csv(t/'all_continual.csv')
        for ds,d in df.groupby('dataset'):
            s=d.groupby('method').forgetting.agg(['mean','std']); fig,ax=plt.subplots(figsize=(6.3,4.1)); ax.bar([METHOD_LABELS.get(i,i) for i in s.index],s['mean'],yerr=s['std']); ax.set_ylabel('Average forgetting'); ax.set_title(f'Continual forgetting — {ds}'); ax.tick_params(axis='x',rotation=20); fig.tight_layout(); fig.savefig(f/f'fig_forgetting_{ds}.png',dpi=300); fig.savefig(f/f'fig_forgetting_{ds}.pdf'); plt.close(fig)
    if (t/'all_continual_stages.csv').exists():
        df=pd.read_csv(t/'all_continual_stages.csv')
        for ds,d in df.groupby('dataset'):
            # Average performance over all tasks observed at each stage.
            q=d.groupby(['method','stage']).macro_f1.mean().reset_index()
            fig,ax=plt.subplots(figsize=(6.3,4.1))
            for m,g in q.groupby('method'):
                ax.plot(g.stage,g.macro_f1,marker='o',label=METHOD_LABELS.get(m,m))
            ax.set_xlabel('Continual stage'); ax.set_ylabel('Mean task Macro-F1'); ax.set_ylim(0,1); ax.set_title(f'Continual adaptation — {ds}'); ax.legend()
            fig.tight_layout(); fig.savefig(f/f'fig_continual_{ds}.png',dpi=300); fig.savefig(f/f'fig_continual_{ds}.pdf'); plt.close(fig)
    if (t/'all_zeroday.csv').exists():
        df=pd.read_csv(t/'all_zeroday.csv'); c=df[df.detector=='combined']
        for ds,d in c.groupby('dataset'):
            s=d.groupby('held_out').auroc.agg(['mean','std']).sort_values('mean'); fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.barh(s.index,s['mean'],xerr=s['std']); ax.set_xlabel('Zero-day AUROC'); ax.set_xlim(0,1); ax.set_title(f'Leave-one-family-out zero-day detection — {ds}'); fig.tight_layout(); fig.savefig(f/f'fig_zeroday_{ds}.png',dpi=300); fig.savefig(f/f'fig_zeroday_{ds}.pdf'); plt.close(fig)
    if (t/'all_fewshot.csv').exists():
        df=pd.read_csv(t/'all_fewshot.csv')
        for ds,d in df.groupby('dataset'):
            fig,ax=plt.subplots(figsize=(6.3,4.1))
            for metric,label in [('new_class_f1','New-class F1'),('historical_macro_f1','Historical Macro-F1')]:
                s=d.groupby('shots')[metric].agg(['mean','std']).sort_index(); ax.errorbar(s.index,s['mean'],yerr=s['std'],marker='o',label=label)
            ax.set_xlabel('Verified zero-day samples'); ax.set_ylabel('Score'); ax.set_ylim(0,1); ax.set_title(f'Few-shot assimilation — {ds}'); ax.legend()
            fig.tight_layout(); fig.savefig(f/f'fig_fewshot_{ds}.png',dpi=300); fig.savefig(f/f'fig_fewshot_{ds}.pdf'); plt.close(fig)
    if (t/'all_explanation.csv').exists():
        df=pd.read_csv(t/'all_explanation.csv')
        for ds,d in df.groupby('dataset'):
            s=d.groupby('method').kendall.agg(['mean','std']); fig,ax=plt.subplots(figsize=(5.5,4.0)); ax.bar([METHOD_LABELS.get(i,i) for i in s.index],s['mean'],yerr=s['std']); ax.set_ylabel('Mean pairwise Kendall τ'); ax.set_title(f'Explanation consistency — {ds}')
            fig.tight_layout(); fig.savefig(f/f'fig_explanation_{ds}.png',dpi=300); fig.savefig(f/f'fig_explanation_{ds}.pdf'); plt.close(fig)
    if (t/'all_drift_series.csv').exists():
        df=pd.read_csv(t/'all_drift_series.csv')
        for ds,d in df.groupby('dataset'):
            q=d.groupby(['detector','window']).score.mean().reset_index(); change=int(d.change_window.median()); fig,ax=plt.subplots(figsize=(6.5,4.1))
            for m,g in q.groupby('detector'): ax.plot(g.window,g.score,marker='o',label=m)
            ax.axhline(1.0,linestyle='--'); ax.axvline(change-0.5,linestyle=':'); ax.set_xlabel('Stream window'); ax.set_ylabel('Normalized drift score'); ax.set_title(f'Real-record drift introduction — {ds}'); ax.legend()
            fig.tight_layout(); fig.savefig(f/f'fig_drift_{ds}.png',dpi=300); fig.savefig(f/f'fig_drift_{ds}.pdf'); plt.close(fig)

    if (t/'all_ablation.csv').exists():
        df=pd.read_csv(t/'all_ablation.csv')
        for ds,d in df.groupby('dataset'):
            s2=d.groupby('variant').forgetting.agg(['mean','std']); fig,ax=plt.subplots(figsize=(7.0,4.2)); ax.bar(s2.index,s2['mean'],yerr=s2['std']); ax.set_ylabel('Average forgetting'); ax.set_title(f'Ablation — {ds}'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(f/f'fig_ablation_{ds}.png',dpi=300); fig.savefig(f/f'fig_ablation_{ds}.pdf'); plt.close(fig)
    if (t/'all_sensitivity.csv').exists():
        df=pd.read_csv(t/'all_sensitivity.csv')
        for (ds,param),d in df.groupby(['dataset','parameter']):
            s2=d.groupby('value').score.agg(['mean','std']).sort_index(); fig,ax=plt.subplots(figsize=(5.8,4.0)); ax.errorbar(s2.index,s2['mean'],yerr=s2['std'],marker='o'); ax.set_xlabel(param); ax.set_ylabel(d.metric.iloc[0]); ax.set_title(f'Sensitivity — {ds}: {param}'); fig.tight_layout(); fig.savefig(f/f'fig_sensitivity_{ds}_{param}.png',dpi=300); fig.savefig(f/f'fig_sensitivity_{ds}_{param}.pdf'); plt.close(fig)
