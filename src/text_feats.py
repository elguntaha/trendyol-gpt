# src/text_feats.py
from __future__ import annotations
import re, hashlib, numpy as np, pandas as pd

TOKEN_SPLIT = re.compile(r"[^\w]+", flags=re.UNICODE)

def _hash_token(tok: str, dim: int) -> int:
    h = hashlib.md5(tok.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % dim

def hashed_bow(series: pd.Series, dim: int = 512, normalize: bool = True, name: str = "cvhash") -> pd.DataFrame:
    """
    Feature hashing (bag-of-words) for comma/space-separated tokens.
    Returns dense DataFrame with columns f"{name}_{i:03d}".
    """
    mat = np.zeros((len(series), dim), dtype="float32")
    for i, text in enumerate(series.fillna("")):
        if not text: 
            continue
        for tok in TOKEN_SPLIT.split(text.lower()):
            if not tok: 
                continue
            j = _hash_token(tok, dim)
            mat[i, j] += 1.0
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True) + 1e-6
        mat = mat / row_sums
    cols = [f"{name}_{i:03d}" for i in range(dim)]
    return pd.DataFrame(mat, columns=cols)

def term_basic_features(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("")
    lens = s.str.len().astype("int32")
    uniq = s.apply(lambda x: len(set(x)))  # unique char count
    spaces = s.str.count(r"\s+").astype("int32")
    return pd.DataFrame({
        "term_len": lens,
        "term_unique_chars": uniq.astype("int32"),
        "term_space_count": spaces,
    })
