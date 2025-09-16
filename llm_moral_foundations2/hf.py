from typing import Optional
from transformers import DynamicCache
import numpy as np

def clone_dynamic_cache(kv_cache, crop: Optional[int]=None):
    if (kv_cache is None) or len(kv_cache)==0:
        return DynamicCache()
    lyrs = kv_cache.to_legacy_cache()
    # 2560, 128, 4096, 1024 8 from attn
    # [layers x (k,v), where
    # k.shape and v.shape [batch, 8=num_heads, seq=623, 128]
    lyrs = ((k.clone()[:, :, :crop], v[:, :, :crop].clone()) for k, v in lyrs)
    lyrs = tuple(lyrs)
    return DynamicCache.from_legacy_cache(lyrs)

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))
