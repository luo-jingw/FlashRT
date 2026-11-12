"""Compare two per-tensor hash files of one safetensors checkpoint (the format of hash_tensors.py: a JSON list of
{name, dtype, shape, sha256}).

Prints the counts of tensors present in both / only in one, and, for the common ones, how many differ in dtype, in
shape and in sha256; then the differing tensors grouped by their name with the layer index replaced by '*'
(for example 'model.language_model.layers.*.mlp.down_proj.weight'), with dtype and shape on both sides.
Writes every differing tensor to OUT_TSV.

usage: python -I compare_tensor_hashes.py A_JSON B_JSON OUT_TSV
"""
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass


@dataclass
class TensorHash:
    name: str
    dtype: str
    shape: list[int]
    sha256: str


@dataclass
class Difference:
    name: str
    a: TensorHash
    b: TensorHash
    fields: list[str]


def load(path: str) -> dict[str, TensorHash]:
    with open(path) as f:
        return {r["name"]: TensorHash(r["name"], r["dtype"], list(r["shape"]), r["sha256"]) for r in json.load(f)}


def differences(a: dict[str, TensorHash], b: dict[str, TensorHash]) -> list[Difference]:
    out = []
    for name in sorted(set(a) & set(b)):
        fields = [k for k in ("dtype", "shape", "sha256") if getattr(a[name], k) != getattr(b[name], k)]
        if fields:
            out.append(Difference(name, a[name], b[name], fields))
    return out


def pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", name)


def main() -> None:
    a_path, b_path, out_tsv = sys.argv[1:4]
    a, b = load(a_path), load(b_path)
    diff = differences(a, b)
    print(f"A {a_path}: {len(a)} tensors; B {b_path}: {len(b)} tensors")
    print(f"common {len(set(a) & set(b))}, only A {len(set(a) - set(b))}, only B {len(set(b) - set(a))}")
    for k in ("dtype", "shape", "sha256"):
        print(f"differ in {k}: {sum(k in d.fields for d in diff)}")
    for name in sorted(set(a) ^ set(b))[:20]:
        print(f"  only {'A' if name in a else 'B'}: {name}")
    groups: Counter[str] = Counter(pattern(d.name) for d in diff)
    example: dict[str, Difference] = {}
    for d in diff:
        example.setdefault(pattern(d.name), d)
    print(f"differing tensors by name pattern ({len(groups)} patterns):")
    for p, n in groups.most_common():
        e = example[p]
        print(f"  {n:5d}  {p}  A {e.a.dtype} {e.a.shape}  B {e.b.dtype} {e.b.shape}  fields {','.join(e.fields)}")
    with open(out_tsv, "w") as f:
        f.write("name\tfields\ta_dtype\ta_shape\ta_sha256\tb_dtype\tb_shape\tb_sha256\n")
        for d in diff:
            f.write(f"{d.name}\t{','.join(d.fields)}\t{d.a.dtype}\t{d.a.shape}\t{d.a.sha256}\t"
                    f"{d.b.dtype}\t{d.b.shape}\t{d.b.sha256}\n")


if __name__ == "__main__":
    main()
