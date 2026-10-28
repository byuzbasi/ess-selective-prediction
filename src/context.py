"""Portable bindings for the unchanged cached-reference comparison arithmetic."""
from common import RUN,STAGE,CFG,atomic,read
SOURCE=RUN
OUT=RUN/'evaluation'
COUNTRIES=CFG['country_order']
SETTINGS=dict(waves=['R7','R11'],models=['catboost','tabm'])
GROUPS=[('all','all'),('education','low'),('education','middle_intermediate'),('education','bachelor_or_higher'),
    ('education','unknown'),('migration','has_background'),('migration','no_background'),('migration','uncertain')]


def group_mask(d,dim,group):
    import numpy as np
    return np.ones(len(d['keys']),dtype=bool) if dim=='all' else d[dim]==group


def aligned(path,keys,fields):
    import numpy as np
    with np.load(path,allow_pickle=False) as f:
        assert set(f.files)=={'keys'}|set(fields)
        assert np.array_equal(f['keys'],keys)
        return {k:f[k] for k in fields}
