"""Worker loader never reads the separately sealed target-test outcomes."""
from common import STAGE, FOLD, ROLES, expected_counts


def load(role,fold=None):
    import numpy as np
    fold=fold or FOLD
    assert fold and role in ROLES
    with np.load(STAGE/'folds'/fold/'payload'/f'{role}.npz',allow_pickle=False) as f:
        p={k:f[k] for k in f.files}
    allowed={'keys','weights','age','cats','dense','ties'} | (set() if role.endswith('_test') else {'y'})
    assert set(p)==allowed
    n=expected_counts(fold)[role]
    assert all(len(a)==n for a in p.values()) and len(set(p['keys']))==n
    assert p['dense'].shape==(n,77) and p['cats'].shape==(n,12)
    assert np.isfinite(p['dense']).all() and (p['weights']>0).all() and abs(p['weights'].sum()-1)<1e-12
    return p
