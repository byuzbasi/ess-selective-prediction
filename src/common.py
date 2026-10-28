"""Portable I/O contracts. Importing does not fit, predict or calibrate."""
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'src'
CFG = json.loads((ROOT / 'config.json').read_text())
RUN = Path(os.environ['ESS_PUBLIC_RUN']).resolve() if 'ESS_PUBLIC_RUN' in os.environ else None
FOLD = os.environ.get('ESS_PUBLIC_FOLD', '')
STAGE = RUN / 'prepared' if RUN else None
PREP = STAGE / 'folds' / FOLD if STAGE and FOLD else STAGE
CURRENT = RUN / 'folds' / FOLD if RUN and FOLD else RUN
ROLES = ['fit','validation','source_calibration','R7_local','R11_local','R7_test','R11_test']
THREADS = ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temp=path.with_name(path.name+'.pending')
    with temp.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def entry(path):
    p=Path(path).resolve()
    # Public manifests contain relative paths, including locally generated outputs.
    anchor=RUN if RUN and p.is_relative_to(RUN) else ROOT
    return dict(path=p.relative_to(anchor).as_posix(), anchor='run' if anchor==RUN else 'repository',
                bytes=p.stat().st_size, sha256=sha(p))


def verify_entries(entries):
    for e in entries:
        root=RUN if e['anchor']=='run' else ROOT
        p=(root / e['path']).resolve()
        assert p.is_relative_to(root) and p.is_file(), e['path']
        assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'], e['path']


def source_entries():
    paths=[ROOT/'run.py',ROOT/'config.json',ROOT/'requirements.txt',ROOT/'data/sources.json']
    paths+=sorted(CODE.glob('*.py'))
    paths+=[ROOT/x['path'] for x in read(ROOT/'data/sources.json')['files']]
    return [entry(p) for p in paths]


def current_signature():
    return hashlib.sha256(json.dumps(source_entries(),sort_keys=True).encode()).hexdigest()


def study_signature():
    return read(RUN/'study.json')['signature']


def signature():
    return hashlib.sha256((study_signature()+'|'+FOLD).encode()).hexdigest()


def config():
    return CFG['model_configuration']


def expected_counts(fold):
    return read(STAGE/'registry.json')['folds'][fold]['counts']


def receipt(path):
    atomic(str(path)+'.receipt.json', dict(signature=signature(),sha256=sha(path)))


def checkpoint_read(path, loader):
    assert read(str(path)+'.receipt.json')==dict(signature=signature(),sha256=sha(path))
    return loader(path)


def assert_environment():
    env=CFG['environment']
    assert platform.python_version()==env['python'], 'Use the recorded Python version'
    for package, version in env['packages'].items():
        assert importlib.metadata.version(package)==version, (package,version)
    assert all(os.environ.get(k)=='2' for k in THREADS), 'Set all five numerical thread variables to 2'


def freeze_check():
    study=read(RUN/'study.json')
    verify_entries(study['inputs'])
    verify_entries(read(STAGE/'manifest.json')['files'])


def authorize(token):
    assert RUN and FOLD in CFG['country_order']
    r=read(RUN/'active_request.json')
    assert os.getppid()==r['controller_pid'] and token==r['token']
    assert r['signature']==study_signature() and r['user_terminal']
    assert_environment()
