"""Validate archive completeness, data boundaries, and all saved prediction errors."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
counts={};max_error=0.;prediction_count=0;data_by_cohort={}
for analysis,expected in [('validation',160),('order_sensitivity',240)]:
    paths=sorted((ROOT/'results'/analysis).glob('*.json'))
    assert len(paths)==expected,(analysis,len(paths),expected)
    records=[json.loads(p.read_text()) for p in paths];counts[analysis]=len(records)
    for r in records:
        assert r['data']['train_sha256']!=r['data']['test_sha256']
        key=(analysis,r['mechanism'],r['case_id'],r['repeat'])
        digests=(r['data']['train_sha256'],r['data']['test_sha256'])
        if key in data_by_cohort:assert data_by_cohort[key]==digests
        data_by_cohort[key]=digests
        tag=f'{analysis}_{r["mechanism"]}_case{r["case_id"]}_rep{r["repeat"]:02d}_q{r["max_order"]}'
        with np.load(ROOT/'results/predictions'/f'{tag}.npz') as arr:
            for arm,m in r['metrics'].items():
                if m is None:continue
                error=float(np.sqrt(np.mean((arr['prediction_'+arm]-arr['truth'])**2)))
                max_error=max(max_error,abs(error-m['rmse']));prediction_count+=1
    if analysis=='validation':assert sum(r['structure']['true_valid'] for r in records)==131
assert max_error<1e-12
main_hashes={h for k,v in data_by_cohort.items() if k[0]=='validation' for h in v}
sens_hashes={h for k,v in data_by_cohort.items() if k[0]=='order_sensitivity' for h in v}
assert not main_hashes & sens_hashes
manifest=ROOT/'MANIFEST_SHA256.json'
if manifest.exists():
    for rel,digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
print(json.dumps({'record_counts':counts,'prediction_errors_recomputed':prediction_count,
                  'max_abs_rmse_difference':max_error,'primary_sensitivity_data_overlap':0,
                  'manifest_checked':manifest.exists()},indent=2))
