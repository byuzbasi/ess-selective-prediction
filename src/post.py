"""Fixed role predictions, six small calibrators, primary80 and source-tau rules."""
import gc
import time
from common import (CURRENT, atomic, read, atomic_npz, signature,
                    checkpoint_read)


def diagnostics(w, inherited=False):
    import numpy as np
    from data import load
    from arithmetic import metric, point_width, standardized, calibration_system, calibration_solve, capacity
    from engine import tree_frame, torch_setup, make_network, torch_load, nn_predict
    roles = ['validation', 'source_calibration', 'R7_local', 'R11_local', 'R7_test', 'R11_test']
    data = {r: load(r) for r in roles}
    cached, timings = {}, {}
    assert not inherited, 'Public runs always start from raw data'
    v = data['validation']
    q = np.array(read(CURRENT / 'reference.json')['quantiles'])
    ref = np.broadcast_to(q, (len(v['keys']), 3))
    reference = dict(common_pinball=metric(v['y'], ref, v['weights']),
        median_mae=float(np.dot(v['weights'], abs(v['y'] - ref[:, 1]))))
    models = {}
    for family in ['catboost', 'tabm']:
        cached[family], timings[family] = {}, {}
        if not inherited:
            if family == 'catboost':
                from catboost import CatBoostRegressor
                net = CatBoostRegressor()
                checkpoint_read(CURRENT / 'catboost/model.cbm', lambda p: net.load_model(str(p)))
                predict = lambda d: net.predict(tree_frame(d), prediction_type='RawFormulaVal', thread_count=2)
            else:
                torch = torch_setup()
                net = make_network(torch)
                net.load_state_dict(torch_load(torch, CURRENT / 'tabm/best.pt')['model'])
                predict = lambda d: nn_predict(torch, net, d)
        for role in roles:
            d = data[role]
            t = time.monotonic()
            assert w.state.get('prediction_calls', 0) < 12
            w.emit(prediction_calls=w.state.get('prediction_calls', 0) + 1)
            p = np.asarray(predict(d), dtype=np.float64)
            atomic_npz(CURRENT / f'{family}_{role}_predictions.npz', keys=d['keys'], predictions=p)
            assert p.shape == (len(d['keys']), 3) and np.isfinite(p).all()
            cached[family][role] = p
            timings[family][role] = dict(seconds=time.monotonic()-t, rows=len(p),
                crossing_fraction=float(np.mean((p[:, 0] > p[:, 1]) | (p[:, 1] > p[:, 2]))),
                outside_0_24_head_fraction=float(np.mean((p < 0) | (p > 24))))
        if not inherited:
            del net, predict
            gc.collect()
    for family in ['catboost', 'tabm']:
        pred = cached[family]
        sc = data['source_calibration']
        f, u = point_width(pred['source_calibration'])
        _, moments = standardized(f, u, weights=sc['weights'])
        beta, residuals = {}, {}
        for role in ['source_calibration', 'R7_local', 'R11_local']:
            assert w.state.get('calibrator_solves_started', 0) < 6
            w.emit(calibrator_solves_started=w.state.get('calibrator_solves_started', 0) + 1)
            d = data[role]
            f, u = point_width(pred[role])
            z, _ = standardized(f, u, moments=moments)
            prior = np.zeros(3) if role == 'source_calibration' else beta['source_calibration']
            a, b = calibration_system(z, d['y'], f, d['weights'], prior)
            beta[role], residuals[role] = calibration_solve(a, b)
            w.emit(calibrator_solves_completed=w.state.get('calibrator_solves_completed', 0) + 1)
            atomic(CURRENT / 'calibrators.partial.json', dict(family=family, role=role,
                   completed=w.state['calibrator_solves_completed']))
        def score(role, coeff):
            f, u = point_width(pred[role])
            z, _ = standardized(f, u, moments=moments)
            return z @ coeff
        source_scores = score('source_calibration', beta['source_calibration'])
        order = np.lexsort((sc['ties'], source_scores))
        tau = float(source_scores[order[np.searchsorted(np.cumsum(sc['weights'][order]), .8, side='left')]])
        w.emit(derived_thresholds=w.state.get('derived_thresholds', 0) + 1)
        target = {}
        for wave in ['R7', 'R11']:
            role, target[wave] = wave + '_test', {}
            d = data[role]
            assert 'y' not in d
            for policy, cal in [('B0', 'source_calibration'), ('B100', wave + '_local')]:
                s = score(role, beta[cal])
                ix, cov = capacity(s, d['weights'], d['ties'])
                primary = np.zeros(len(s), dtype=bool)
                primary[ix] = True
                fixed_tau = s <= tau
                atomic_npz(CURRENT / f'{family}_{wave}_{policy}_capacity.npz', keys=d['keys'],
                    accepted_primary=primary, accepted_tau=fixed_tau)
                target[wave][policy] = dict(primary=cov, fixed_tau_coverage=float(np.dot(d['weights'], fixed_tau)))
        p = pred['validation']
        ix, cov = capacity(score('validation', beta['source_calibration']), v['weights'], v['ties'])
        selected_mae = float(np.sum(v['weights'][ix] * abs(v['y'][ix] - p[ix, 1])) / v['weights'][ix].sum())
        models[family] = dict(common_pinball=metric(v['y'], p, v['weights']),
            median_mae=float(np.dot(v['weights'], abs(v['y'] - p[:, 1]))),
            selective_validation_MAE=selected_mae, validation_capacity=cov,
            source_calibration_moments=moments, calibration_coefficients={r: a.tolist() for r,a in beta.items()},
            calibration_residuals=residuals, source_tau=tau, target_X_capacity=target)
    atomic(CURRENT / 'diagnostics.json', dict(signature=signature(), inherited=inherited,
        reference=reference, models=models, timings=timings, target_test_labels_used=False,
        target_group_findings=False, validation_selection_bias=True, cap_limited_flags_preserved=True))
    w.emit(success=True)
