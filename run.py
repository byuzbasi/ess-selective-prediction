"""Portable entry point. Help, check, prepare and verify never train models."""
import argparse
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['check','prepare','run','resume','verify','evaluate'])
    ap.add_argument('--run',default='results/run01',help='New or existing directory inside results/')
    ap.add_argument('--user-terminal',action='store_true',help='Required explicit acknowledgement for training')
    args=ap.parse_args()
    if args.command in ['run','resume'] and not args.user_terminal:
        ap.error('Training requires --user-terminal; check and prepare are fit-free')
    path=Path(args.run)
    run=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if not run.is_relative_to(ROOT/'results'):
        ap.error('--run must be inside this repository results/ directory')
    os.environ['ESS_PUBLIC_RUN']=str(run)
    os.environ.pop('ESS_PUBLIC_FOLD',None)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        if name in os.environ and os.environ[name]!='2':
            ap.error(name+' must equal 2')
        os.environ[name]='2'
    sys.path.insert(0,str(ROOT/'src'))
    from common import assert_environment,freeze_check
    assert_environment()
    if args.command=='check':
        from selfcheck import check
        check()
    elif args.command=='prepare':
        from controller import prepare_run
        prepare_run()
        print('Prepared raw-data cohorts, roles and payloads; no fit or training.')
    elif args.command in ['run','resume']:
        from controller import run as execute
        execute(resume=args.command=='resume')
    elif args.command=='verify':
        from controller import verify_saved
        verify_saved();print('All 19 model and decision checkpoints verified.')
    elif args.command=='evaluate':
        from evaluate import evaluate
        evaluate(run,run/'evaluation')
        from report import report
        report()


if __name__=='__main__':
    main()
