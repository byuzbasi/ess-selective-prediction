"""Short, deterministic checks. Learner imports and calibration solves are blocked."""
import ast
import importlib.abc
import sys
from common import ROOT,CFG,read,sha


class BlockLearners(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ['torch','catboost','tabm','rtdl_revisiting_models']:
            raise RuntimeError('Fit-free check blocked learner import: '+fullname)


def check():
    guard=BlockLearners();sys.meta_path.insert(0,guard)
    import numpy as np
    from arithmetic import capacity,metric
    for p in [ROOT/'run.py',*sorted((ROOT/'src').glob('*.py'))]:
        ast.parse(p.read_text(),filename=p.name)
    for item in read(ROOT/'data/sources.json')['files']:
        assert sha(ROOT/item['path'])==item['sha256']
    assert len(CFG['country_order'])==len(set(CFG['country_order']))==19
    assert CFG['seeds']==dict(source_roles=2026102601,local_reserve=2026102602,score_ties=2026102603,model=2026102604)
    assert len(CFG['features'])==13
    assert not set(f['name'] for f in CFG['features']) & set(CFG['excluded_as_inputs'])
    selected,info=capacity(np.array([0.,0.,1.]),np.array([.4,.3,.3]),np.array(['b','a','c']))
    assert selected.tolist()==[1,0] and abs(info['coverage']-.7)<1e-15
    assert metric(np.array([1.]),np.array([[1.,1.,1.]]),np.array([1.]))==0
    import engine,post,data,worker,evaluate,baseline,report
    assert not any(m in sys.modules for m in ['torch','catboost','tabm'])
    print('PASS: syntax, raw SHA-256, 19-country configuration, seeds, 13 inputs, arithmetic and import guards; zero fitting.',flush=True)
