from __future__ import annotations
import json, platform, sys
from pathlib import Path
import numpy, pandas, sklearn, scipy, matplotlib, torch
info={
 'python':sys.version,
 'platform':platform.platform(),
 'processor':platform.processor(),
 'machine':platform.machine(),
 'torch':torch.__version__,
 'cuda_available':torch.cuda.is_available(),
 'cuda_version':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'numpy':numpy.__version__, 'pandas':pandas.__version__, 'scikit_learn':sklearn.__version__, 'scipy':scipy.__version__, 'matplotlib':matplotlib.__version__,
}
out=Path(__file__).resolve().parents[1]/'results/environment.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(info,indent=2)); print(json.dumps(info,indent=2))
