"""Frozen pilot numerical functions, with fold payloads and single logging option.
Training/loss/architecture code is preserved; execution is terminal-worker guarded.
"""
import os
import random
import time
from common import PREP, ROLES, atomic, read, config, assert_environment, signature, receipt, checkpoint_read

def torch_setup():
    import numpy as np
    import torch
    seed = config()['common_fit_policy']['master_model_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch



def make_network(torch):
    from tabm import EnsembleView, make_tabm_backbone, LinearEnsemble
    c = config()['tabm_proposed']
    chunks = read(PREP / 'payload/contract.json')['chunks']
    return torch.nn.Sequential(EnsembleView(k=c['k']), make_tabm_backbone(
        d_in=77, n_blocks=c['n_blocks'], d_block=c['d_block'], dropout=c['dropout'],
        activation=c['activation'], k=c['k'], arch_type=c['arch_type'],
        start_scaling_init=c['start_scaling_init'], start_scaling_init_chunks=chunks),
        LinearEnsemble(c['d_block'], 3, k=c['k'])).cpu()



def torch_save(torch, path, obj):
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    receipt(path)



def torch_load(torch, path):
    return checkpoint_read(path, lambda p: torch.load(p, map_location='cpu', weights_only=False))



def nn_predict(torch, net, data):
    import numpy as np
    net.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(data['keys']), 512):
            p = net(torch.from_numpy(data['dense'][i:i + 512]))
            assert p.ndim == 3 and p.shape[1:] == (16, 3)
            out.append(p.mean(1).double().numpy())
    return np.concatenate(out)



def tree_frame(data):
    import pandas as pd
    names = read(PREP / 'payload/contract.json')['cat_names']
    f = pd.DataFrame(data['cats'], columns=names)
    f.insert(0, 'agea', data['age'])
    return f



def reference(w):
    import numpy as np
    from data import load
    assert_environment()
    # Validate all seven schemas before the first estimation work unit.
    for role in ROLES:
        load(role)
    fit = load('fit')
    w.emit(reference=1, estimation_started='constant_reference')
    order = np.argsort(fit['y'], kind='stable')
    cdf = np.cumsum(fit['weights'][order])
    q = fit['y'][order[np.searchsorted(cdf, [.1, .5, .9], side='left')]]
    atomic(w.run / 'reference.json', dict(quantiles=q.tolist(), signature=signature(),
           n=len(fit['y']), source_fit_only=True, empirical_quantile_rule='smallest Y at CDF>=q'))
    w.emit(completed=['constant_reference'], success=True)



def catboost(w):
    import numpy as np
    from catboost import CatBoostRegressor, Pool
    from data import load
    assert_environment()
    c = config()['catboost_proposed']
    a, b = load('fit'), load('validation')
    train = Pool(tree_frame(a), a['y'], cat_features=list(range(1, 13)), weight=len(a['y']) * a['weights'])
    valid = Pool(tree_frame(b), b['y'], cat_features=list(range(1, 13)), weight=len(b['y']) * b['weights'])
    unit = w.run / 'catboost'
    unit.mkdir(exist_ok=False)
    model = CatBoostRegressor(**c['constructor_parameters'], train_dir=str(unit), logging_level='Silent')
    times = []

    class Observer:
        def after_iteration(self, info):
            times.append(time.monotonic() - w.start)
            w.emit(iteration=int(info.iteration), iteration_elapsed_seconds=times,
                   checkpoints=int((unit / 'snapshot.cbsnapshot').exists()))
            return True

    w.emit(training_started=True, load_import_seconds=time.monotonic() - w.start)
    began = time.monotonic()
    model.fit(train, eval_set=valid, use_best_model=True, early_stopping_rounds=30,
              save_snapshot=True, snapshot_interval=30, snapshot_file=str(unit / 'snapshot.cbsnapshot'),
              callbacks=[Observer()])
    train_seconds = time.monotonic() - began
    assert model.tree_count_ > 0 and len(times) <= 400
    # Required first-family technical gate, before the controller may start TabM.
    gate_started = time.monotonic()
    gate = np.asarray(model.predict(valid, prediction_type='RawFormulaVal', thread_count=2), dtype=np.float64)
    assert gate.shape == (len(b['keys']), 3) and np.isfinite(gate).all()
    gate_seconds = time.monotonic() - gate_started
    tmp = unit / 'model.cbm.pending'
    checkpoint_start = time.monotonic()
    model.save_model(str(tmp), format='cbm')
    tmp.replace(unit / 'model.cbm')
    receipt(unit / 'model.cbm')
    atomic(unit / 'result.json', dict(signature=signature(), tree_count=model.tree_count_,
           iterations_executed=len(times), best_iteration=model.get_best_iteration(),
           cap_limited=len(times) >= 400, train_seconds=train_seconds,
           first_family_prediction_gate_pass=True, gate_seconds=gate_seconds,
           checkpoint_seconds=time.monotonic() - checkpoint_start,
           native_best_validation=model.get_best_score(), training_starts=1))
    w.emit(success=True, checkpoints=w.state['checkpoints'] + 1, completed=['catboost'])



def tabular(w):
    import numpy as np
    from data import load
    from arithmetic import metric
    torch = torch_setup()
    assert_environment()
    c = config()['tabm_proposed']
    a, v = load('fit'), load('validation')
    net = make_network(torch)
    params = dict(c['optimizer'])
    params.pop('class')
    params['betas'] = tuple(params['betas'])
    optimizer = torch.optim.AdamW(net.parameters(), **params)
    generator = torch.Generator().manual_seed(2026102604)
    x, y, weight = torch.from_numpy(a['dense']), torch.from_numpy(a['y']), torch.from_numpy(a['weights'])
    qs = torch.tensor([.1, .5, .9], dtype=torch.float64)
    unit = w.run / 'tabm'
    unit.mkdir(exist_ok=False)
    best, patience_anchor, stale, updates, history = float('inf'), float('inf'), 0, 0, []
    best_state, best_epoch = None, None
    w.emit(training_started=True, load_import_seconds=time.monotonic() - w.start)
    for epoch in range(1, 31):
        t = time.monotonic()
        net.train()
        order = torch.randperm(len(y), generator=generator)
        for ix in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            prediction = net(x[ix])
            assert prediction.shape == (len(ix), 16, 3)
            residual = y[ix, None, None] - prediction.double()
            losses = residual * (qs - (residual < 0).double())
            loss = len(y) / len(ix) * (weight[ix] * losses.sum(-1).mean(-1)).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite TabM loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite TabM gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in net.parameters()):
                raise FloatingPointError('Nonfinite TabM parameter')
            updates += 1
            assert updates <= 1260
            w.emit(epoch=epoch, updates=updates)
        training_seconds = time.monotonic() - t
        t = time.monotonic()
        prediction = nn_predict(torch, net, v)
        score = metric(v['y'], prediction, v['weights'])
        validation_seconds = time.monotonic() - t
        assert np.isfinite(score)
        meaningful_improvement = score < patience_anchor * (1 - 1e-4)
        if meaningful_improvement:
            patience_anchor, stale = score, 0
        else:
            stale += 1
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
            best_state = {k: p.detach().cpu().clone() for k, p in net.state_dict().items()}
            torch_save(torch, unit / 'best.pt', dict(signature=signature(), model=best_state, best_epoch=epoch))
        t = time.monotonic()
        history.append(dict(epoch=epoch, updates=updates, train_seconds=training_seconds,
                            validation_seconds=validation_seconds, common_pinball=score, improved=improved))
        torch_save(torch, unit / 'epoch.pt', dict(signature=signature(), model=net.state_dict(),
                   optimizer=optimizer.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(), shuffle_rng=generator.get_state(),
                   epoch=epoch, updates=updates, best=best, best_model=best_state, best_epoch=best_epoch,
                   stale=stale, patience_anchor=patience_anchor, history=history, training_starts=1))
        history[-1]['checkpoint_seconds'] = time.monotonic() - t
        atomic(unit / 'epochs.json', history)
        w.emit(epoch=epoch, updates=updates, checkpoints=epoch, best=best)
        if stale >= 5:
            break
    # Best checkpoint reader is exercised only during the terminal pilot.
    saved = torch_load(torch, unit / 'best.pt')
    assert saved['signature'] == signature() and saved['best_epoch'] == best_epoch
    atomic(unit / 'result.json', dict(signature=signature(), epochs=epoch, updates=updates,
           best_epoch=best_epoch, best_common_pinball=best, cap_limited=epoch >= 30,
           patience_triggered=stale >= 5, training_starts=1, history=history))
    w.emit(success=True, completed=['tabm'])


