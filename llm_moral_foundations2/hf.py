from transformers import DynamicCache
import numpy as np

def clone_dynamic_cache(kv_cache):
    if (kv_cache is None) or len(kv_cache)==0:
        return DynamicCache()
    c = kv_cache.to_legacy_cache()
    c = ((a.clone(), b.clone()) for a, b in c)
    c = tuple(c)
    return DynamicCache.from_legacy_cache(c)

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))
