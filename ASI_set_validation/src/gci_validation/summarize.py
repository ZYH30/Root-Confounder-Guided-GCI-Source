"""Build publication tables from all prescribed final-setting records."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
MECHANISMS=['additive_sine','multiplicative','heteroscedastic','interaction']

def build_tables(root:Path, output:Path):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; effects=[]; checks=[]
    for analysis in ['validation','order_sensitivity']:
        for path in sorted((root/analysis).glob('*.json')):
            r=json.loads(path.read_text());s=r['structure'];z=s['constructor']['selected']
            common={k:r[k] for k in ['analysis','mechanism','case_id','repeat','max_order']}
            row={**common,'valid':int(s['true_valid']),'precision':s['precision'],'recall':s['recall'],'f1':s['f1'],
                 'set_size':len(z) if z is not None else np.nan,'descendant':int(bool(s['true_forbidden'])),
                 'other_invalid':int(not s['true_valid'] and not s['true_forbidden']),
                 'construction_failure':int(z is None),'ci_calls':r['counts']['conditional_tests']}
            rows.append(row)
            tag=f'{analysis}_{r["mechanism"]}_case{r["case_id"]}_rep{r["repeat"]:02d}_q{r["max_order"]}'
            predictions=root/'predictions'/f'{tag}.npz'
            arrays=np.load(predictions) if predictions.exists() else None
            for arm,m in r['metrics'].items():
                if m is None:continue
                effects.append({**common,'arm':arm,**m})
                if arrays is not None and 'prediction_'+arm in arrays:
                    calc=float(np.sqrt(np.mean((arrays['prediction_'+arm]-arrays['truth'])**2)))
                    checks.append(abs(calc-m['rmse']))
    if not rows:raise ValueError(f'No records in {root}')
    frame=pd.DataFrame(rows);effect=pd.DataFrame(effects)
    frame.to_csv(output/'individual_structure_metrics.csv',index=False)
    effect.to_csv(output/'individual_effect_metrics.csv',index=False)
    main=frame[frame.analysis=='validation']
    def summarize_structure(g):
        return dict(n=len(g),valid_count=int(g.valid.sum()),validity=float(g.valid.mean()),
                    precision=float(g.precision.mean()),recall=float(g.recall.mean()),f1=float(g.f1.mean()),
                    mean_size=float(g.set_size.mean()),descendant=int(g.descendant.sum()),
                    other_invalid=int(g.other_invalid.sum()),construction_failures=int(g.construction_failure.sum()))
    summary=[];cell_summary=[]
    for m in MECHANISMS:
        g=main[main.mechanism==m]
        if g.empty:continue
        summary.append({'mechanism':m,**summarize_structure(g)})
        for c in sorted(g.case_id.unique()):cell_summary.append({'mechanism':m,'case_id':int(c),**summarize_structure(g[g.case_id==c])})
    pd.DataFrame(summary).to_csv(output/'final_set_recovery.csv',index=False)
    pd.DataFrame(cell_summary).to_csv(output/'graph_specific_structure.csv',index=False)
    em=effect[effect.analysis=='validation']
    em.groupby(['mechanism','arm'],sort=False).rmse.agg(['mean','std','size']).reset_index().to_csv(output/'downstream_effects.csv',index=False)
    em.groupby(['mechanism','case_id','arm'],sort=False).rmse.agg(['mean','std','size']).reset_index().to_csv(output/'graph_specific_effects.csv',index=False)
    sens=[]
    for q in [2,3,4]:
        g=frame[(frame.analysis=='order_sensitivity')&(frame.max_order==q)]
        e=effect[(effect.analysis=='order_sensitivity')&(effect.max_order==q)&(effect.arm=='gci')]
        if g.empty:continue
        sens.append({'max_order':q,**summarize_structure(g),'rmse_mean':float(e.rmse.mean()),
                     'rmse_sd':float(e.rmse.std()),'mean_ci_calls':float(g.ci_calls.mean())})
    pd.DataFrame(sens).to_csv(output/'conditioning_order_sensitivity.csv',index=False)
    report={'structure_records':len(frame),'effect_records':len(effect),
            'validation_records':len(main),'validation_valid_count':int(main.valid.sum()),
            'sensitivity_records':int((frame.analysis=='order_sensitivity').sum()),
            'prediction_errors_recomputed':len(checks),'maximum_rmse_recalculation_difference':max(checks,default=None)}
    (output/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('results'));p.add_argument('--output',type=Path,default=Path('results/tables'));a=p.parse_args();build_tables(a.input,a.output)
if __name__=='__main__':main()
