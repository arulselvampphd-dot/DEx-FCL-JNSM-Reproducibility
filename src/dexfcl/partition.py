from __future__ import annotations
from pathlib import Path
import numpy as np
from .utils import ensure_dir, save_json

def dirichlet_partition(indices: np.ndarray, y: np.ndarray, n_clients:int, alpha:float, seed:int, min_size:int=16,
                        max_tries:int=200) -> list[np.ndarray]:
    rng=np.random.default_rng(seed); classes=np.unique(y[indices])
    for _ in range(max_tries):
        clients=[[] for _ in range(n_clients)]
        for c in classes:
            cidx=indices[y[indices]==c].copy(); rng.shuffle(cidx)
            p=rng.dirichlet(np.full(n_clients,alpha))
            cuts=(np.cumsum(p)*len(cidx)).astype(int)[:-1]
            for k,part in enumerate(np.split(cidx,cuts)): clients[k].extend(part.tolist())
        arr=[np.array(v,dtype=np.int64) for v in clients]
        if min(map(len,arr))>=min_size:
            for a in arr: rng.shuffle(a)
            return arr
    raise RuntimeError(f"Could not generate Dirichlet partition with min_size={min_size}; increase alpha or data size.")

def save_partition_set(root:str|Path,dataset:str,seed:int,alpha:float,train_parts,val_parts,y,groups):
    out=ensure_dir(Path(root)/dataset/f"seed_{seed}"/f"alpha_{alpha:g}")
    manifest={"dataset":dataset,"seed":seed,"alpha":alpha,"clients":[]}
    for k,(tr,va) in enumerate(zip(train_parts,val_parts)):
        np.save(out/f"client_{k:02d}_train.npy",tr); np.save(out/f"client_{k:02d}_val.npy",va)
        manifest["clients"].append({"client":k,"n_train":len(tr),"n_val":len(va),
            "train_counts":{groups[int(c)]:int((y[tr]==c).sum()) for c in np.unique(y[tr])},
            "val_counts":{groups[int(c)]:int((y[va]==c).sum()) for c in np.unique(y[va])}})
    save_json(manifest,out/'manifest.json'); return out
