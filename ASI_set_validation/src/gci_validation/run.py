"""Reproduce the final validation or the matched conditioning-order analysis."""
from __future__ import annotations
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
import argparse,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import networkx as nx
from .asi import ASIConfig, discover
from .adjustment import graph_from_parents,select_adjustment,adjustment_validity,recovery
from .graph_cases import CASE_GRAPHS
from .data_generators import GeneratorConfig,generate_case_dataset
from .estimation import estimate_effect

PACKAGE_ROOT=Path(__file__).resolve().parents[2]

def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def run_one(analysis: str, mechanism: str, case_id: int, repeat: int,
            max_order: int, output: Path, settings: dict) -> dict:
    tag=f'{analysis}_{mechanism}_case{case_id}_rep{repeat:02d}_q{max_order}'
    path=output/analysis/f'{tag}.json'
    if path.exists():
        raise FileExistsError(f'{path} already exists; select an empty output directory')
    started=time.perf_counter()
    protocol=settings[analysis]
    seed=protocol['base_seed']+protocol['seed_stride']*repeat
    test_seed=seed+settings['test_seed_offset']
    train=generate_case_dataset(case_id,mechanism,GeneratorConfig(n_samples=settings['n_train'],random_seed=seed))
    test=generate_case_dataset(case_id,mechanism,GeneratorConfig(n_samples=settings['n_test'],random_seed=test_seed))
    case=CASE_GRAPHS[case_id]
    config=ASIConfig(**settings['asi'],max_condition_set=max_order)
    discovery,audit=discover(train,case.observed_covariates,config)
    estimated=graph_from_parents(discovery.parent_dict,case.observed_covariates)
    constructor=select_adjustment(estimated,case.observed_covariates)
    true=nx.DiGraph(case.edges);true.add_nodes_from(case.nodes)
    reference=select_adjustment(true,case.observed_covariates)['selected']
    selected=constructor['selected']
    validity=adjustment_validity(true,selected) if selected is not None else {'valid':False,'forbidden':[],'backdoor_blocked':False}
    recovered=recovery(reference,selected)
    # Exact equality is not a primary endpoint because admissible sets can differ.
    recovered.pop('exact_match',None)
    structure={'constructor':constructor,'reference_set':reference,
               'true_valid':bool(validity['valid']),'true_forbidden':validity['forbidden'],
               'true_backdoor_blocked':bool(validity['backdoor_blocked']),**recovered}
    arms={'gci':selected,'true_graph_gci':reference,'all_covariates':list(case.observed_covariates),'no_adjustment':[]}
    if analysis == 'order_sensitivity': arms = {'gci':selected}
    metrics={};arrays={'truth':(test.y-test.y_delta).to_numpy()}
    for arm,variables in arms.items():
        metric,prediction=estimate_effect(train,test,variables)
        metrics[arm]=metric
        if prediction is not None:arrays[f'prediction_{arm}']=prediction
    (output/'predictions').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions'/f'{tag}.npz',**arrays)
    def digest(frame):return hashlib.sha256(frame.to_numpy(dtype='<f8').tobytes()).hexdigest()
    result={'analysis':analysis,'mechanism':mechanism,'case_id':case_id,'repeat':repeat,'max_order':max_order,
            'train_seed':seed,'test_seed':test_seed,'configuration':asdict(config),
            'data':{'n_train':len(train),'n_test':len(test),'train_sha256':digest(train),'test_sha256':digest(test)},
            'structure':structure,'discovery':asdict(discovery),'metrics':metrics,
            'counts':{'conditional_tests':len(audit['ci_tests'])+len(audit['local_ci_guard_tests']),
                      'screening_and_source_tests':len(audit['ci_tests']),
                      'local_guard_tests':len(audit['local_ci_guard_tests'])},
            'audit':audit,'elapsed_seconds':time.perf_counter()-started}
    atomic_json(path,result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',choices=['validation','order_sensitivity'],default='validation')
    parser.add_argument('--config',type=Path,default=PACKAGE_ROOT/'configs/final_settings.json')
    parser.add_argument('--output',type=Path,default=PACKAGE_ROOT/'reproduced')
    parser.add_argument('--case',type=int,choices=[1,2,3,4],nargs='+')
    parser.add_argument('--mechanism',nargs='+',choices=['additive_sine','multiplicative','heteroscedastic','interaction'])
    parser.add_argument('--repeats',type=int,help='Run the first n prescribed repeats (default: full analysis)')
    parser.add_argument('--order',type=int,choices=[2,3,4],nargs='+',help='Subset of the prescribed orders')
    args=parser.parse_args();settings=json.loads(args.config.read_text())
    protocol=settings[args.analysis];repeats=args.repeats if args.repeats is not None else protocol['repeats']
    if repeats<1 or repeats>protocol['repeats']:parser.error('repeats is outside the prescribed range')
    orders=args.order or protocol['max_orders']
    if not set(orders)<=set(protocol['max_orders']):parser.error('order is not part of the selected analysis')
    jobs=[(m,c,r,q) for m in (args.mechanism or settings['mechanisms'])
          for c in (args.case or settings['case_ids']) for r in range(repeats) for q in orders]
    for index,(m,c,r,q) in enumerate(jobs,1):
        row=run_one(args.analysis,m,c,r,q,args.output,settings)
        error=row['metrics']['gci']
        print(f'{index}/{len(jobs)} {m} case={c} repeat={r} q={q} valid={row["structure"]["true_valid"]} rmse={error["rmse"] if error else None}',flush=True)
    print('Completed. Use gci_validation.summarize to build tables.',flush=True)
if __name__=='__main__':main()
