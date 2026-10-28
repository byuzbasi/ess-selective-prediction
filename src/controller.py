"""One controller, two numerical threads, verified whole-country checkpoints."""
import os
import secrets
import sys
import time
from common import (ROOT,CODE,RUN,STAGE,CFG,read,atomic,sha,entry,verify_entries,
                    now,source_entries,current_signature,study_signature,assert_environment,freeze_check)
from runtime import supervise,progress


def verify_country(country):
    folder=RUN/'folds'/country
    manifest=read(folder/'manifest.json')
    assert manifest['signature']==study_signature() and manifest['country']==country
    verify_entries(manifest['files'])
    states=[read(folder/(phase+'.state.json')) for phase in ['reference','catboost','tabm','diagnostics']]
    assert all(s['success'] for s in states)
    diag=states[-1]
    assert diag['calibrator_solves_started']==diag['calibrator_solves_completed']==6
    assert diag['prediction_calls']==12 and diag['derived_thresholds']==2
    assert read(folder/'catboost/result.json')['training_starts']==1
    assert read(folder/'tabm/result.json')['training_starts']==1
    assert not read(folder/'diagnostics.json')['target_test_labels_used']
    return manifest


def verify_saved(run=None):
    if run is not None:
        assert run.resolve()==RUN
    freeze_check()
    for c in CFG['country_order']:
        verify_country(c)
    completion=read(RUN/'completion.json')
    assert completion['countries']==19 and completion['estimation_units']==171
    verify_entries(completion['country_manifests'])


def prepare_run():
    from prepare import prepare
    assert not RUN.exists(), 'Existing run: use resume or a new directory; never overwrite'
    assert RUN.is_relative_to(ROOT/'results'), 'Run directories must be inside results/'
    RUN.mkdir(parents=True,exist_ok=False)
    atomic(RUN/'study.json',dict(signature=current_signature(),inputs=source_entries(),
        created=now(),software=CFG['environment'],country_order=CFG['country_order']))
    registry=prepare()
    (RUN/'folds').mkdir()
    return registry


def run(resume=False):
    assert_environment()
    if not RUN.exists():
        assert not resume, 'Cannot resume a missing run'
        prepare_run()
    freeze_check()
    assert current_signature()==study_signature(), 'Code, configuration or raw data changed'
    began=time.monotonic()
    if (RUN/'ledger.json').exists():
        assert resume, 'Existing execution: use resume; run never overwrites it'
        ledger=read(RUN/'ledger.json')
        assert ledger['signature']==study_signature()
    else:
        assert not resume, 'Use run for a prepared but unstarted directory'
        ledger=dict(signature=study_signature(),created=now(),deadline_unix=time.time()+CFG['limits']['wall_seconds'])
        atomic(RUN/'ledger.json',ledger)
    deadline=began+max(0.,ledger['deadline_unix']-time.time())
    caps=CFG['limits']['phase_seconds']
    limits=dict(stop_grace_seconds_within_each_phase=5,
        RSS_sampled_parent_plus_worker_threshold_bytes=CFG['limits']['rss_bytes'],
        output_all_including_pending_bytes_max=CFG['limits']['output_bytes'],
        logs_traces_bytes_max=CFG['limits']['logs_bytes'])
    state=dict(run_id=RUN.name,status='RUNNING',completed=0,running=0,failed=0,pending=19,
        durably_committed_countries=0,main_completed=0,calibrators_completed=0,reference_completed=0,
        peak_rss_bytes=0,phase='starting',last_completed_unit='',ETA_seconds=None,
        ETA_scope='Unknown until complete-country timings are available',memory_is_hard_OS_cap=False)
    last_print=0.
    def tick(extra=None,force=False):
        nonlocal last_print
        if extra:
            extra=dict(extra)
            state['peak_rss_bytes']=max(state['peak_rss_bytes'],extra.pop('peak_rss_bytes',0))
            state.update(extra)
        at=time.monotonic()
        if force or at-last_print>=30:
            state.update(heartbeat=now(),elapsed=at-began,remaining_wall_seconds=max(0.,deadline-at),
                task_percent=100*state['completed']/19,percentage_kind='validated completed countries')
            progress(RUN,state);last_print=at
    token=secrets.token_hex(24)
    atomic(RUN/'active_request.json',dict(token=token,controller_pid=os.getpid(),user_terminal=True,signature=study_signature()))
    completed=[];durations=[]
    for country in CFG['country_order']:
        folder=RUN/'folds'/country
        if (folder/'manifest.json').exists():
            verify_country(country)
            completed.append(country)
            state.update(completed=len(completed),pending=19-len(completed),durably_committed_countries=len(completed),
                main_completed=2*len(completed),calibrators_completed=6*len(completed),reference_completed=len(completed))
            continue
        # An incomplete country is preserved, and never silently refitted on resume.
        assert not folder.exists(), f'Incomplete country {country}: retained; automatic retry is prohibited'
        assert time.monotonic()<deadline-5, 'Original run deadline expired'
        folder.mkdir()
        fold_start=time.monotonic()
        state.update(fold=country,running=1,phase='reference')
        os.environ['ESS_PUBLIC_FOLD']=country
        for phase in ['reference','catboost','tabm','diagnostics']:
            state['phase']=phase;tick(force=True)
            command=[sys.executable,'-B',str(CODE/'worker.py'),phase,'--token',token]
            result=supervise(command,folder,phase,caps[phase],deadline,limits,tick)
            if result['reason']!='completed':
                state.update(status='STOPPED',running=0,failed=1,stop_reason=result['reason'])
                tick(force=True)
                raise RuntimeError(f'{country}/{phase}: '+result['reason'])
            assert read(folder/(phase+'.state.json'))['success']
        files=[entry(p) for p in sorted(folder.rglob('*')) if p.is_file()]
        atomic(folder/'manifest.json',dict(signature=study_signature(),country=country,files=files))
        verify_country(country)
        completed.append(country);durations.append(time.monotonic()-fold_start)
        state.update(completed=len(completed),running=0,pending=19-len(completed),durably_committed_countries=len(completed),
            main_completed=2*len(completed),calibrators_completed=6*len(completed),reference_completed=len(completed),
            last_completed_unit=country,ETA_seconds=sum(durations)/len(durations)*(19-len(completed)),
            ETA_scope='Remaining model and decision country stages; final verification and evaluation additional')
        tick(force=True)
    atomic(RUN/'completion.json',dict(signature=study_signature(),countries=19,estimation_units=171,
        country_manifests=[entry(RUN/'folds'/c/'manifest.json') for c in CFG['country_order']],
        completed=now(),target_test_labels_used=False))
    verify_saved()
    state.update(status='MODELS_AND_DECISIONS_VERIFIED',phase='complete');tick(force=True)
    print('Verified 19 countries. Run evaluate separately to access target-test outcomes.',flush=True)
