from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,matthews_corrcoef,roc_auc_score,average_precision_score,roc_curve

def multiclass_metrics(y_true,y_pred,probs=None,n_classes=None):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    out={"accuracy":accuracy_score(y_true,y_pred),"precision":p,"recall":r,"macro_f1":f,"mcc":matthews_corrcoef(y_true,y_pred)}
    if probs is not None:
        try:
            labels=np.arange(probs.shape[1]); one=np.eye(probs.shape[1])[y_true]
            out['auroc']=roc_auc_score(one,probs,multi_class='ovr',average='macro',labels=labels)
        except Exception: out['auroc']=float('nan')
    return out

def binary_open_metrics(y_unknown,score,threshold):
    pred=(score>threshold).astype(int)
    p,r,f,_=precision_recall_fscore_support(y_unknown,pred,average='binary',zero_division=0)
    try: auroc=roc_auc_score(y_unknown,score); aupr=average_precision_score(y_unknown,score)
    except Exception: auroc=aupr=float('nan')
    known=(y_unknown==0); fpr=float(pred[known].mean()) if known.any() else float('nan')
    fpr95=float('nan')
    try:
        fpr_arr,tpr,_=roc_curve(y_unknown,score); fpr95=float(fpr_arr[np.where(tpr>=.95)[0][0]])
    except Exception: pass
    return {"auroc":auroc,"aupr":aupr,"unknown_precision":p,"unknown_recall":r,"unknown_f1":f,"fpr":fpr,"fpr95":fpr95}
