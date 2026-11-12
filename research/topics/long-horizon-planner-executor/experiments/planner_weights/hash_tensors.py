import hashlib, json, sys
import torch
from safetensors import safe_open
P, OUT = sys.argv[1], sys.argv[2]
rows = []
with safe_open(P, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        b = t.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        rows.append({"name": k, "dtype": str(t.dtype).replace("torch.", ""),
                     "shape": list(t.shape), "sha256": hashlib.sha256(b).hexdigest()})
rows.sort(key=lambda r: r["name"])
json.dump(rows, open(OUT, "w"), indent=1)
print(len(rows), "tensors ->", OUT)
