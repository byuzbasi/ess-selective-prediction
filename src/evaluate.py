"""Original descriptive metrics, with portable paths and no estimation."""
from pathlib import Path
from common import STAGE, CFG, read, atomic, atomic_npz, entry, verify_entries, study_signature
from controller import verify_saved
def stats(y,p,a,chosen,domain):
    import numpy as np
    ix=domain;den=float(a[ix].sum());accepted=ix & chosen
    selected=float(a[accepted].sum())
    return dict(n=int(ix.sum()),weight_mass=den,accepted_n=int(accepted.sum()),accepted_weight_mass=selected,
        coverage=None if den==0 else selected/den,
        full_MAE=None if den==0 else float(np.dot(a[ix],abs(y[ix]-p[ix]))/den),
        accepted_MAE=None if selected==0 else float(np.dot(a[accepted],abs(y[accepted]-p[accepted]))/selected),
        kish=None if den==0 else den*den/float(np.dot(a[ix],a[ix])),
        low_precision=bool(ix.sum()<100 or (den>0 and den*den/np.dot(a[ix],a[ix])<50)))


def evaluate(run, output):
    import numpy as np
    run, output=Path(run).resolve(),Path(output).resolve()
    verify_saved(run)
    assert not output.exists() and output == run / 'evaluation'
    output.mkdir(exist_ok=False)
    rows=[]
    for country in CFG['country_order']:
        folder=run/'folds'/country
        for wave in ['R7','R11']:
            with np.load(STAGE/'evaluation'/f'{country}_{wave}.npz',allow_pickle=False) as file:
                d={k:file[k] for k in file.files}
            domains={'main':np.ones(len(d['keys']),bool),'age25_64':(d['age']>=25)&(d['age']<=64),
                     'education_code5_with_BA':np.ones(len(d['keys']),bool)}
            if wave=='R11':domains['mode1']=d['mode']==1
            for family in ['catboost','tabm']:
                with np.load(folder/f'{family}_{wave}_test_predictions.npz',allow_pickle=False) as f:
                    assert np.array_equal(f['keys'],d['keys']);point=f['predictions'][:,1]
                decisions={('full','full'):np.ones(len(point),bool)}
                for budget in ['B0','B100']:
                    with np.load(folder/f'{family}_{wave}_{budget}_capacity.npz',allow_pickle=False) as f:
                        assert np.array_equal(f['keys'],d['keys'])
                        decisions[('primary80',budget)]=f['accepted_primary']
                        decisions[('source_tau',budget)]=f['accepted_tau']
                for domain_name,domain in domains.items():
                    education=d['education'].copy()
                    if domain_name=='education_code5_with_BA':education[d['eisced']==5]='bachelor_or_higher'
                    groups=[('all','all',np.ones(len(point),bool))]
                    groups += [('education',g,education==g) for g in ['low','middle_intermediate','bachelor_or_higher','unknown']]
                    groups += [('migration',g,d['migration']==g) for g in ['has_background','no_background','uncertain']]
                    groups += [('intersection',e+'|'+m,(education==e)&(d['migration']==m))
                        for e in ['low','middle_intermediate','bachelor_or_higher'] for m in ['has_background','no_background']]
                    for (rule,budget),mask in decisions.items():
                        for dimension,group,gmask in groups:
                            row=dict(country=country,wave=wave,family=family,domain=domain_name,
                                rule=rule,budget=budget,dimension=dimension,group=group)
                            row.update(stats(d['y'],point,d['anweight'],mask,domain&gmask));rows.append(row)
    # Each country metric is computed first; group gaps never mix country compositions.
    macro=[];changes=[]
    for family in ['catboost','tabm']:
        for wave in ['R7','R11']:
            for domain in sorted({r['domain'] for r in rows if r['wave']==wave}):
                for rule,budget in [('full','full'),('primary80','B0'),('primary80','B100'),('source_tau','B0'),('source_tau','B100')]:
                    selected=[r for r in rows if (r['family'],r['wave'],r['domain'],r['rule'],r['budget'])==(family,wave,domain,rule,budget)]
                    for dimension,group in sorted({(r['dimension'],r['group']) for r in selected}):
                        cells=[r for r in selected if (r['dimension'],r['group'])==(dimension,group)]
                        record=dict(family=family,wave=wave,domain=domain,rule=rule,budget=budget,dimension=dimension,group=group)
                        for metric in ['coverage','full_MAE','accepted_MAE']:
                            usable=[r for r in cells if r[metric] is not None]
                            record[metric]=None if not usable else float(np.mean([r[metric] for r in usable]))
                            record[metric+'_countries']=[r['country'] for r in usable]
                            den=sum(r['weight_mass'] if metric!='accepted_MAE' else r['accepted_weight_mass'] for r in usable)
                            record['population_weighted_'+metric]=None if den==0 else sum(r[metric]*(r['weight_mass'] if metric!='accepted_MAE' else r['accepted_weight_mass']) for r in usable)/den
                        macro.append(record)
                    for dim,low,high in [('education','low','bachelor_or_higher'),('migration','has_background','no_background')]:
                        gaps={}
                        for c in CFG['country_order']:
                            pair={r['group']:r for r in selected if r['country']==c and r['dimension']==dim}
                            if pair[low]['coverage'] is not None and pair[high]['coverage'] is not None:
                                gaps[c]=pair[low]['coverage']-pair[high]['coverage']
                        macro.append(dict(kind='group_coverage_gap',family=family,wave=wave,domain=domain,rule=rule,budget=budget,
                            dimension=dim,gap_definition=low+' minus '+high,countries=list(gaps),country_gaps=gaps,
                            equal_country_mean_gap=None if not gaps else float(np.mean(list(gaps.values())))))
    gaps=[r for r in macro if r.get('kind')=='group_coverage_gap']
    for a in gaps:
        if a['budget']!='B0':continue
        b=next(x for x in gaps if (x['family'],x['wave'],x['domain'],x['rule'],x['dimension'],x['budget'])==
            (a['family'],a['wave'],a['domain'],a['rule'],a['dimension'],'B100'))
        pair=[c for c in CFG['country_order'] if c in a['country_gaps'] and c in b['country_gaps']]
        changes.append(dict(kind='B100_minus_B0_signed_gap_change',family=a['family'],wave=a['wave'],domain=a['domain'],rule=a['rule'],
            dimension=a['dimension'],countries=pair,mean_change=None if not pair else float(np.mean([b['country_gaps'][c]-a['country_gaps'][c] for c in pair]))))
    for a in gaps:
        if a['wave']!='R7' or a['domain']=='mode1':continue
        candidates=[b for b in gaps if (b['family'],b['wave'],b['domain'],b['rule'],b['dimension'],b['budget'])==
            (a['family'],'R11',a['domain'],a['rule'],a['dimension'],a['budget'])]
        if not candidates:continue
        b=candidates[0];pair=[c for c in CFG['country_order'] if c in a['country_gaps'] and c in b['country_gaps']]
        changes.append(dict(kind='R11_minus_R7_signed_gap_change',family=a['family'],domain=a['domain'],rule=a['rule'],budget=a['budget'],
            dimension=a['dimension'],countries=pair,mean_change=None if not pair else float(np.mean([b['country_gaps'][c]-a['country_gaps'][c] for c in pair]))))
    # Pair conditional-error comparisons on the same estimable countries.
    index={(r['family'],r['wave'],r['domain'],r['rule'],r['budget'],r['dimension'],r['group'],r['country']):r for r in rows}
    for a in macro:
        if a.get('kind') or a['budget']!='B0':continue
        key=(a['family'],a['wave'],a['domain'],a['rule'],a['dimension'],a['group'])
        b=next(x for x in macro if not x.get('kind') and
            (x['family'],x['wave'],x['domain'],x['rule'],x['dimension'],x['group'],x['budget'])==key+('B100',))
        countries=[c for c in CFG['country_order'] if c in a['accepted_MAE_countries'] and c in b['accepted_MAE_countries']]
        differences=[index[key[:4]+('B100',)+key[4:]+(c,)]['accepted_MAE']-
                     index[key[:4]+('B0',)+key[4:]+(c,)]['accepted_MAE'] for c in countries]
        changes.append(dict(kind='B100_minus_B0_accepted_MAE',family=a['family'],wave=a['wave'],domain=a['domain'],
            rule=a['rule'],dimension=a['dimension'],group=a['group'],countries=countries,
            mean_change=None if not differences else float(np.mean(differences))))
    atomic(output/'country_group_metrics.json',rows)
    atomic(output/'macro_metrics.json',macro)
    atomic(output/'paired_contrasts.json',changes)
    atomic(output/'evaluation_record.json',dict(signature=study_signature(),source_run_manifest_sha256=__import__('common').sha(run/'completion.json'),
        countries=19,training_or_calibration=0,target_test_labels_read=True,predictions_and_decisions_changed=False,
        CI=False,p_values=False,SDDF_R7_verified=False,scope='descriptive point estimates; fixed19 countries; folds not independent repetitions',
        primary_target_cells=152,sensitivity_rules_frozen=True))
    atomic(output/'manifest.json',dict(files=[entry(p) for p in sorted(output.iterdir()) if p.is_file()],
        source_run_manifest=entry(run/'completion.json')))
    verify_entries(read(output/'manifest.json')['files'])
    print('Point-estimate evaluation complete; no training or calibration.',flush=True)

