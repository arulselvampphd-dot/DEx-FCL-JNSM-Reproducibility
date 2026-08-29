from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .mappings import map_ciciot_label, map_edge_label, CICIOT_GROUPS, EDGE_GROUPS
from .utils import save_json

@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    fine: np.ndarray
    groups: list[str]
    feature_names: list[str]

    @classmethod
    def load(cls, npz_path: str | Path) -> "DatasetBundle":
        z = np.load(npz_path, allow_pickle=True)
        return cls(z["X"].astype(np.float32), z["y"].astype(np.int64), z["fine"].astype(str),
                   z["groups"].astype(str).tolist(), z["feature_names"].astype(str).tolist())

def _clean_numeric_frame(df: pd.DataFrame, label_cols: set[str], drop_patterns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    drops = set(label_cols)
    for c in df.columns:
        lc = c.lower().strip()
        if any(re.search(p, lc) for p in drop_patterns): drops.add(c)
    x = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore")
    # Keep numeric columns only to avoid high-cardinality identifiers and payload leakage.
    x = x.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, x.columns.tolist()

def _two_pass_sample(files: list[Path], label_col: str, mapper, cap_per_group: int, chunksize: int,
                     read_kwargs: dict, seed: int, drop_patterns: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    counts: dict[str, int] = {}
    fine_counts: dict[str, int] = {}
    print("[prepare] pass 1/2: counting labels")
    for f in tqdm(files):
        for ch in pd.read_csv(f, usecols=[label_col], chunksize=chunksize, low_memory=False, **read_kwargs):
            vals = ch[label_col].astype(str)
            for fine, n in vals.value_counts().items():
                fine_counts[fine] = fine_counts.get(fine, 0) + int(n)
                g = mapper(fine); counts[g] = counts.get(g, 0) + int(n)
    probs = {g: min(1.0, cap_per_group / max(1, n)) for g, n in counts.items()}
    rng = np.random.default_rng(seed)
    frames=[]; groups=[]; fines=[]
    feature_names=None
    print("[prepare] pass 2/2: deterministic class-capped sampling")
    for f in tqdm(files):
        for ch in pd.read_csv(f, chunksize=chunksize, low_memory=False, **read_kwargs):
            raw_fine = ch[label_col].astype(str).to_numpy()
            mapped = np.array([mapper(v) for v in raw_fine], dtype=object)
            keep = np.zeros(len(ch), dtype=bool)
            for g, p in probs.items():
                idx = np.flatnonzero(mapped == g)
                if len(idx): keep[idx] = rng.random(len(idx)) < p
            if not keep.any(): continue
            sub = ch.loc[keep].reset_index(drop=True)
            sub_fine = raw_fine[keep]; sub_group = mapped[keep]
            x, names = _clean_numeric_frame(sub, {label_col, "Attack_label", "Attack_type", "label"}, drop_patterns)
            if feature_names is None: feature_names = names
            else:
                # Align columns if CSV partitions vary slightly.
                x = x.reindex(columns=feature_names, fill_value=0)
            frames.append(x.astype(np.float32)); groups.append(sub_group); fines.append(sub_fine)
    if not frames: raise RuntimeError("No rows sampled; verify dataset path and label column.")
    X = pd.concat(frames, ignore_index=True)
    G = np.concatenate(groups); F = np.concatenate(fines)
    # Exact cap after Bernoulli sampling.
    keep_all=[]
    rng2=np.random.default_rng(seed+991)
    for g in sorted(set(G.tolist())):
        idx=np.flatnonzero(G==g)
        if len(idx)>cap_per_group: idx=rng2.choice(idx, cap_per_group, replace=False)
        keep_all.append(np.asarray(idx))
    keep=np.concatenate(keep_all); rng2.shuffle(keep)
    X=X.iloc[keep].reset_index(drop=True); G=G[keep]; F=F[keep]
    return X, G, F

def prepare_ciciot2023(input_path: str | Path, output_npz: str | Path, cap_per_group: int=120_000,
                       chunksize: int=250_000, seed: int=2026) -> dict:
    p=Path(input_path)
    files=sorted(p.glob("part-*.csv")) if p.is_dir() else [p]
    if not files: raise FileNotFoundError(f"No CICIoT2023 part-*.csv files under {p}")
    label_col="label"
    Xdf, G, F = _two_pass_sample(files,label_col,map_ciciot_label,cap_per_group,chunksize,{},seed,
        drop_patterns=[r"(^|\.)src(_|\.|$)",r"(^|\.)dst(_|\.|$)",r"payload",r"timestamp",r"frame\.time",r"mac"])
    groups=CICIOT_GROUPS
    y=np.array([groups.index(g) for g in G], dtype=np.int64)
    X=Xdf.to_numpy(np.float32)
    out=Path(output_npz); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fine=F.astype(str), groups=np.array(groups), feature_names=np.array(Xdf.columns.astype(str)))
    meta={"dataset":"ciciot2023","rows":int(len(y)),"features":int(X.shape[1]),"groups":groups,
          "group_counts":{g:int((G==g).sum()) for g in groups},"source_files":len(files),"cap_per_group":cap_per_group,
          "sampling_seed":seed}
    save_json(meta,out.with_suffix('.json')); return meta

def prepare_edgeiiotset(input_csv: str | Path, output_npz: str | Path, cap_per_group: int=120_000,
                        chunksize: int=200_000, seed: int=2026) -> dict:
    p=Path(input_csv)
    if p.is_dir():
        cand=list(p.rglob("DNN-EdgeIIoT-dataset.csv"))
        if not cand: cand=list(p.rglob("ML-EdgeIIoT-dataset.csv"))
        if not cand: raise FileNotFoundError(f"DNN/ML Edge-IIoTset CSV not found under {p}")
        p=cand[0]
    # Conservative leakage controls: identifiers, literal payload/content and raw endpoint fields are excluded.
    drop=[r"frame\.time",r"ip\.src",r"ip\.dst",r"src_host",r"dst_host",r"eth\.src",r"eth\.dst",r"payload",
          r"http\.file_data",r"full_uri",r"uri\.query",r"mqtt\.msg$",r"arp\.src",r"arp\.dst"]
    Xdf,G,F=_two_pass_sample([p],"Attack_type",map_edge_label,cap_per_group,chunksize,{},seed,drop)
    groups=EDGE_GROUPS; y=np.array([groups.index(g) for g in G],dtype=np.int64); X=Xdf.to_numpy(np.float32)
    out=Path(output_npz); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,X=X,y=y,fine=F.astype(str),groups=np.array(groups),feature_names=np.array(Xdf.columns.astype(str)))
    meta={"dataset":"edgeiiotset","rows":int(len(y)),"features":int(X.shape[1]),"groups":groups,
          "group_counts":{g:int((G==g).sum()) for g in groups},"source_file":str(p),"cap_per_group":cap_per_group,
          "sampling_seed":seed,"feature_policy":"numeric-only with identifier/payload exclusions"}
    save_json(meta,out.with_suffix('.json')); return meta

def stratified_split_scale(bundle: DatasetBundle, seed:int, train_frac=.70, val_frac=.10):
    idx=np.arange(len(bundle.y))
    train_idx,tmp_idx=train_test_split(idx,test_size=1-train_frac,stratify=bundle.y,random_state=seed)
    rel_test=(1-train_frac-val_frac)/(1-train_frac)
    val_idx,test_idx=train_test_split(tmp_idx,test_size=rel_test,stratify=bundle.y[tmp_idx],random_state=seed+1)
    scaler=StandardScaler().fit(bundle.X[train_idx])
    X=scaler.transform(bundle.X).astype(np.float32)
    return X, train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64), scaler
