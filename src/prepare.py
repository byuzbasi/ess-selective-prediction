"""Deterministic raw-data preparation; no model or calibrator is imported."""
import math
import numpy as np
import pandas as pd
from common import ROOT, RUN, STAGE, CFG, ROLES, atomic, atomic_npz, read, sha, entry
from roles import KEYS, canonical_features, roles_for_wave, role_mask, transform_inputs


def cohorts():
    """Match the recorded R preparation, preserving original Stata row identities."""
    for src in read(ROOT/'data/sources.json')['files']:
        assert sha(ROOT/src['path'])==src['sha256'], src['path']
    result, features, roles = {}, {}, {}
    items=CFG['outcome']['negative_items']+CFG['outcome']['positive_items']
    born=CFG['migration']['variables']
    raw_features=[f['source'] for f in CFG['features'] if f['source']!='existing approved cohort']
    needed=list(dict.fromkeys(['cntry','idno','essround','edition','agea','eisced',
        'anweight','pspwght','dweight','pweight']+born+items+raw_features))
    for wave,file,edition in [('R7','ESS7e02_3.dta','2.3'),('R11','ESS11e04_2.dta','4.2')]:
        d=pd.read_stata(ROOT/'data'/file,columns=needed+(['mode'] if wave=='R11' else []),convert_categoricals=False)
        assert d.edition.eq(edition).all() and d.essround.eq(int(wave[1:])).all()
        assert not d.duplicated(['cntry','idno']).any()
        scope=d.cntry.isin(CFG['country_order'])
        age_ok=d.agea.between(25,89)
        all8=d[items].isin([1,2,3,4]).all(axis=1)
        selected=scope & age_ok & all8
        for name in ['anweight','pspwght','dweight','pweight']:
            assert np.isfinite(d.loc[scope,name]).all() and d.loc[scope,name].gt(0).all()
        assert np.allclose(d.loc[scope,'anweight'],d.loc[scope,'pspwght']*d.loc[scope,'pweight'],rtol=1e-6,atol=1e-12)
        if wave=='R11':
            assert d.loc[scope,'mode'].dropna().isin([1,2]).all()
        raw=d.loc[selected].reset_index(drop=True)
        c=raw.copy()
        c['raw_row_index']=np.flatnonzero(selected)+1
        c['education_group']='unknown'
        for name,codes in CFG['education']['groups'].items():
            c.loc[c.eisced.isin(codes),'education_group']=name
        c['migration_group']=np.where(c[born].eq(2).any(axis=1),'has_background',
            np.where(c[born].eq(1).all(axis=1),'no_background','uncertain'))
        c['CESD8_observed_sum']=(c[CFG['outcome']['negative_items']]-1).sum(axis=1)+(4-c[CFG['outcome']['positive_items']]).sum(axis=1)
        assert c.CESD8_observed_sum.between(0,24).all()
        assert len(c)=={'R7':31319,'R11':26677}[wave]
        c['w_country_equal_cohort']=c.anweight/c.anweight.groupby(c.cntry).transform('sum')/19
        f=canonical_features(raw,c,CFG)
        r=roles_for_wave(c,wave,CFG)
        assert len(f.columns)==17 and not f.isna().any().any()
        assert int(r.target_role.eq('local_reserve').sum())==1900
        result[wave],features[wave],roles[wave]=c,f,r
    return result,features,roles


def metadata(cohort, features, roles):
    meta=dict(feature_names=[f['name'] for f in CFG['features']],folds={})
    for fold in CFG['country_order']:
        mask=role_mask(roles,fold,'fit')
        w=roles.loc[mask,'source_within_weight'].to_numpy(float)/18
        age=features.loc[mask,'agea'].to_numpy(float)
        mu=float(np.dot(w,age)); variance=float(np.dot(w,np.square(age-mu)))
        meta['folds'][fold]=dict(age_source_fit_moments=dict(mean=mu,variance_population=variance,
            scale=math.sqrt(variance) if variance>0 else 1.),categorical_vocabularies={})
        for name in meta['feature_names'][1:]:
            meta['folds'][fold]['categorical_vocabularies'][name]=dict(observed_in_source_fit=sorted(features.loc[mask,name].unique().tolist()))
    return meta


def key_strings(frame):
    return np.array(['|'.join([str(int(r)),c,str(int(i)),str(int(j))])
        for r,c,i,j in frame[KEYS].itertuples(index=False,name=None)])


def dense(x,fold,meta):
    record=meta['folds'][fold];a=record['age_source_fit_moments']
    blocks=[((x.agea.to_numpy(float)-a['mean'])/a['scale'])[:,None]]
    for col,vocab in record['categorical_vocabularies'].items():
        blocks.append((x[col].to_numpy()[:,None]==np.array(vocab['observed_in_source_fit'])[None,:]).astype(float))
    result=np.concatenate(blocks,axis=1).astype(np.float32)
    assert result.shape==(len(x),77) and np.isfinite(result).all()
    return result


def payload(c,f,r,fold,role,meta):
    named='local_reserve' if role.endswith('_local') else ('test' if role.endswith('_test') else role)
    mask=role_mask(r,fold,named)
    x=transform_inputs(f.loc[mask],meta,fold)
    cats=list(meta['folds'][fold]['categorical_vocabularies'])
    w=r.loc[mask,'source_within_weight'].to_numpy(float)/18 if role in ROLES[:3] else r.loc[mask,'target_within_weight'].to_numpy(float)
    assert abs(w.sum()-1)<1e-12 and (w>0).all()
    pack=dict(keys=key_strings(f.loc[mask]),weights=w,age=x.agea.to_numpy(float),
        cats=x[cats].to_numpy(str),dense=dense(x,fold,meta),ties=r.loc[mask,'score_ties_hash'].to_numpy(str))
    if not role.endswith('_test'):
        pack['y']=c.loc[mask,'CESD8_observed_sum'].to_numpy(float)
    return pack,mask


def prepare():
    assert STAGE and not STAGE.exists(), 'Refuse to overwrite prepared inputs'
    cs,fs,rs=cohorts();meta=metadata(cs['R7'],fs['R7'],rs['R7'])
    STAGE.mkdir()
    (STAGE/'evaluation').mkdir()
    registry=dict(folds={})
    for fold in CFG['country_order']:
        folder=STAGE/'folds'/fold/'payload';folder.mkdir(parents=True)
        cats=list(meta['folds'][fold]['categorical_vocabularies'])
        chunks=[1]+[len(v['observed_in_source_fit']) for v in meta['folds'][fold]['categorical_vocabularies'].values()]
        counts={}
        for role in ROLES:
            wave='R11' if role.startswith('R11') else 'R7'
            c=cs[wave];pack,mask=payload(c,fs[wave],rs[wave],fold,role,meta)
            atomic_npz(folder/f'{role}.npz',**pack);counts[role]=len(pack['keys'])
            if role.endswith('_test'):
                truth=dict(keys=pack['keys'],y=c.loc[mask,'CESD8_observed_sum'].to_numpy(float),
                    weights=pack['weights'],anweight=c.loc[mask,'anweight'].to_numpy(float),
                    age=c.loc[mask,'agea'].to_numpy(float),eisced=c.loc[mask,'eisced'].to_numpy(float),
                    education=c.loc[mask,'education_group'].to_numpy(str),migration=c.loc[mask,'migration_group'].to_numpy(str),
                    mode=c.loc[mask,'mode'].to_numpy(float) if wave=='R11' else np.full(len(pack['keys']),np.nan))
                atomic_npz(STAGE/'evaluation'/f'{fold}_{wave}.npz',**truth)
        atomic(folder/'contract.json',dict(chunks=chunks,cat_names=cats,counts=counts))
        registry['folds'][fold]=dict(counts=counts,fit_seed=2026102604)
    atomic(STAGE/'registry.json',registry)
    atomic(STAGE/'metadata.json',meta)
    files=[entry(p) for p in sorted(STAGE.rglob('*')) if p.is_file()]
    atomic(STAGE/'manifest.json',dict(files=files))
    return registry
