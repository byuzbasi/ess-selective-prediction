"""Write descriptive CSV tables and base-R scientific figure source inputs."""
import csv
import math
from common import RUN,CFG,read,atomic,entry


def csv_file(path,rows,fields):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def report():
    from baseline import calculate
    counts=calculate()
    out=RUN/'evaluation'
    macro=read(out/'macro_metrics.json');country=read(out/'country_group_metrics.json')
    baseline=read(out/'baseline_macro.json');turnover=read(out/'turnover_macro.json')
    clean=[r for r in macro if not r.get('kind')]
    # Tables expose unrounded numbers; the manuscript supplies narrative interpretation.
    csv_file(out/'table1_error_and_coverage.csv',
        [r for r in clean if r['domain']=='main' and r['dimension']=='all'],
        ['family','wave','rule','budget','coverage','full_MAE','accepted_MAE'])
    gaps=[r for r in macro if r.get('kind')=='group_coverage_gap' and r['domain']=='main']
    csv_file(out/'table2_group_coverage.csv',gaps,
        ['family','wave','rule','budget','dimension','gap_definition','equal_country_mean_gap'])
    changes=read(out/'paired_contrasts.json')
    csv_file(out/'table3_adaptation.csv',[r for r in changes if r['domain']=='main'],
        ['kind','family','wave','rule','dimension','group','mean_change'])
    csv_file(out/'baseline_comparison.csv',baseline,
        ['wave','family','selection','dimension','group','K','model_MAE','reference_MAE','delta_MAE'])
    csv_file(out/'selection_turnover.csv',turnover,
        ['wave','family','rule','dimension','group','K','pooled_n','pooled_turnover_n','turnover','coverage_change'])
    gap_plot=[]
    for r in gaps:
        if r['wave']=='R11' and r['rule']=='primary80' and r['budget']=='B0' and r['dimension']=='education':
            gap_plot += [dict(country=c,family=r['family'],gap_pp=100*v) for c,v in r['country_gaps'].items()]
    csv_file(out/'figure1_country_gaps.csv',gap_plot,['country','family','gap_pp'])
    csv_file(out/'figure2_turnover.csv',
        [dict(family=r['family'],rule=r['rule'],turnover_percent=100*r['turnover']) for r in turnover
         if r['wave']=='R11' and r['dimension']=='all'],['family','rule','turnover_percent'])
    atomic(out/'descriptive_report.json',dict(accepted=True,**counts,
        confidence_intervals=False,p_values=False,bootstrap=False,
        estimand='Equal mean of within-country anweight-weighted metrics; paired estimable countries for gaps.',
        limits=['Observed complete-case CES-D8 score, not a clinical diagnosis.',
                'No population significance, causal discrimination or equivalence claim.',
                'One fixed reserve; constant source-median reference only.',
                'Source-validation selection and cross-country measurement limits remain.']))
    atomic(out/'report_manifest.json',dict(files=[entry(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print('Descriptive tables and figure data generated. No confidence intervals or p-values.',flush=True)
