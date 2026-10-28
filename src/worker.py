"""Controller-owned workers; direct estimation calls are rejected."""
import argparse
import time
from common import CURRENT, FOLD, atomic, authorize, signature, freeze_check, now


class Worker:
    def __init__(self,phase):
        self.phase,self.run,self.began=phase,CURRENT,time.monotonic()
        self.start=self.began
        self.state=dict(phase=phase,fold=FOLD,signature=signature(),success=False,
            reference=0,calibrator_solves_started=0,calibrator_solves_completed=0,
            prediction_calls=0,derived_thresholds=0,checkpoints=0,epoch=0,updates=0,iteration=0)
        self.emit()

    def emit(self,**kw):
        self.state.update(kw)
        self.state.update(heartbeat=now(),elapsed=time.monotonic()-self.began)
        atomic(CURRENT/(self.phase+'.state.json'),self.state)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['reference','catboost','tabm','diagnostics'])
    ap.add_argument('--token',required=True)
    args=ap.parse_args()
    authorize(args.token)
    freeze_check()
    w=Worker(args.phase)
    try:
        if args.phase=='diagnostics':
            from post import diagnostics
            diagnostics(w)
        else:
            from engine import reference,catboost,tabular
            dict(reference=reference,catboost=catboost,tabm=tabular)[args.phase](w)
    except BaseException as exc:
        w.emit(error=type(exc).__name__+': '+str(exc),success=False)
        raise
