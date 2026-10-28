"""New comparisons using saved numbers and decisions only; no estimation."""
import math
from collections import defaultdict
import numpy as np
from context import *

def turnover(a,b,w,group):
    n=int(group.sum());mass=float(w[group].sum())
    result=dict(n=n,weight_mass=mass)
    for label,mask in [('00',~a&~b),('01',~a&b),('10',a&~b),('11',a&b)]:
        chosen=group&mask
        result['n'+label]=int(chosen.sum());result['w'+label]=float(w[chosen].sum())
        result['share'+label]=None if mass==0 else result['w'+label]/mass
    result['turnover_n']=result['n01']+result['n10']
    result['turnover']=None if mass==0 else (result['w01']+result['w10'])/mass
    result['coverage_change']=None if mass==0 else (result['w01']-result['w10'])/mass
    return result

def calculate():
    from evaluate import stats  # Retained, validated within-country arithmetic.
    baseline=[];changes=[];counts={'R7':0,'R11':0}
    for country in COUNTRIES:
        folder=SOURCE/'folds'/country;ref=read(folder/'reference.json')
        assert ref['source_fit_only'] and len(ref['quantiles'])==3
        q=ref['quantiles'];assert all(math.isfinite(x) for x in q) and q==sorted(q)
        for wave in SETTINGS['waves']:
            with np.load(STAGE/'evaluation'/f'{country}_{wave}.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
            n=len(d['keys']);counts[wave]+=n
            assert len(set(d['keys']))==n and np.isfinite(d['y']).all() and (d['anweight']>0).all()
            constant=np.full(n,q[1]) # Evaluates the saved constant; no fitted model is loaded.
            for family in SETTINGS['models']:
                p=aligned(folder/f'{family}_{wave}_test_predictions.npz',d['keys'],['predictions'])['predictions']
                assert p.shape==(n,3) and np.isfinite(p).all()
                masks={b:aligned(folder/f'{family}_{wave}_{b}_capacity.npz',d['keys'],['accepted_primary','accepted_tau']) for b in ['B0','B100']}
                for v in masks.values():assert all(a.dtype==np.bool_ and a.shape==(n,) for a in v.values())
                for selection,mask in [('full',np.ones(n,bool)),('B0',masks['B0']['accepted_primary']),('B100',masks['B100']['accepted_primary'])]:
                    for dim,group in GROUPS:
                        g=group_mask(d,dim,group)
                        m=stats(d['y'],p[:,1],d['anweight'],mask,g)
                        r=stats(d['y'],constant,d['anweight'],mask,g)
                        assert m['accepted_n']==r['accepted_n'] and m['accepted_weight_mass']==r['accepted_weight_mass']
                        mm,rm=m['accepted_MAE'],r['accepted_MAE']
                        baseline.append(dict(country=country,wave=wave,family=family,selection=selection,dimension=dim,group=group,
                            saved_reference_median=q[1],group_n=m['n'],n=m['accepted_n'],weight_mass=m['accepted_weight_mass'],
                            model_MAE=mm,reference_MAE=rm,delta_MAE=None if mm is None else mm-rm,
                            low_precision=m['low_precision']))
                for rule,field in [('primary80','accepted_primary'),('source_tau','accepted_tau')]:
                    for dim,group in GROUPS:
                        changes.append(dict(country=country,wave=wave,family=family,rule=rule,dimension=dim,group=group,
                            **turnover(masks['B0'][field],masks['B100'][field],d['anweight'],group_mask(d,dim,group))))
    assert counts=={'R7':29419,'R11':24777}
    atomic(OUT/'baseline_country.json',baseline);atomic(OUT/'turnover_country.json',changes)
    bmacro=[];tmacro=[]
    for rows,fields,metrics,result in [
        (baseline,['wave','family','selection','dimension','group'],['model_MAE','reference_MAE','delta_MAE'],bmacro),
        (changes,['wave','family','rule','dimension','group'],['share00','share01','share10','share11','turnover','coverage_change'],tmacro)]:
        buckets=defaultdict(list)
        for r in rows:buckets[tuple(r[k] for k in fields)].append(r)
        for key,rr in buckets.items():
            usable=[r for r in rr if r[metrics[0]] is not None]
            assert all(all(r[k] is not None for k in metrics) for r in usable)
            record=dict(zip(fields,key));record.update(K=len(usable),countries=[r['country'] for r in usable],pooled_n=sum(r['n'] for r in rr))
            if rows is changes:record['pooled_turnover_n']=sum(r['turnover_n'] for r in rr)
            for k in metrics:record[k]=None if not usable else math.fsum(r[k] for r in usable)/len(usable)
            result.append(record)
    atomic(OUT/'baseline_macro.json',bmacro);atomic(OUT/'turnover_macro.json',tmacro)
    assert [len(x) for x in [baseline,bmacro,changes,tmacro]]==[1824,96,1216,64]
    return dict(people=counts,evaluation_units=76,baseline_country_rows=len(baseline),baseline_macro_rows=len(bmacro),
        turnover_country_rows=len(changes),turnover_macro_rows=len(tmacro))
