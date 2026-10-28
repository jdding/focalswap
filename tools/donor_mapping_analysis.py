#!/usr/bin/env python3
"""Analyze every predeclared map/model/scope separately on fixed support."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from control_ladder_analysis import load_seed,cluster_bootstrap,interval_status,sha256_file

SEEDS=(20230601,20230602,20230603)
MAPS=(20260921,20260922,20260923,20260924,20260925)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--mapping-manifest',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run=a.run_root;out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    reference=json.loads(a.reference_manifest.read_text())
    expected={x['mapping_seed']:x for x in json.loads(a.mapping_manifest.read_text())['mapping_records']}
    records=[];support_identity={};hashes={}
    for mapping in MAPS:
        for index,seed in enumerate(SEEDS,1):
            frame,integrity,_=load_seed(run/f'mapping_{mapping}/scores',a.baseline_root,seed,reference)
            if integrity['intervention_seed']!=mapping or integrity['category_donor_map']['donor_map_sha256']!=expected[mapping]['donor_map_sha256']:
                raise ValueError('wrong mapping identity')
            hashes[f'{mapping}:{seed}']=integrity['output_csv_sha256']
            for scope,n,clusters in [('latest_1',87044,2843),('all',66139,2170)]:
                sub=frame.loc[frame[f'category_{scope}_eligible'].astype(bool)]
                if len(sub)!=n or sub.user_id.nunique()!=clusters:raise ValueError('support denominator changed')
                keys=sub[['official_row_index','user_id']].to_numpy()
                if scope in support_identity and not np.array_equal(support_identity[scope],keys):raise ValueError('common rows changed')
                support_identity[scope]=keys
                full=sub.full_rr.to_numpy(float);c2=sub[f'category_{scope}_rr'].to_numpy(float)
                c0=sub[f'neutral_{scope}_rr'].to_numpy(float)
                users=sub.user_id.to_numpy()
                record={'mapping_seed':mapping,'checkpoint':f'UniSAR S{index}','model_seed':seed,'scope':scope,
                        'rows':n,'user_clusters':clusters,'c0_delta_rr':float((full-c0).mean()),
                        'half_tie':float(((full>c2)+.5*(full==c2)).mean()),
                        'target_rank_changed_fraction':float((full!=c2).mean()),
                        'donor_map_sha256':integrity['category_donor_map']['donor_map_sha256'],
                        'ci_level':.975,'bootstrap_iters':10000}
                for position,(name,values) in enumerate([('c2',full-c2),('paired_c0_minus_c2',c2-c0)]):
                    bootseed=int.from_bytes(hashlib.sha256(f'R266:{mapping}:{seed}:{scope}:{position}'.encode()).digest()[:4],'little')
                    point,low,high,count=cluster_bootstrap(values,users,10000,bootseed,alpha=.025)
                    assert count==clusters
                    record.update({name+'_delta_rr':point,name+'_ci_low':low,name+'_ci_high':high,
                                   name+'_status':interval_status(low,high),name+'_bootstrap_seed':bootseed})
                records.append(record)
            print(f'ANALYZED map={mapping} model={seed}',flush=True)
    table=pd.DataFrame(records).sort_values(['scope','model_seed','mapping_seed'])
    if len(table)!=30:raise ValueError('incomplete mapping/model/scope matrix')
    all_source=table[table.scope=='all']
    summary={'schema':'focalswap.r266.mapping_sensitivity.v1','status':'COMPLETE','rows':records,
             'formal_score_cells':15,'scope_readouts':30,'mapping_seeds':list(MAPS),'model_seeds':list(SEEDS),
             'all_source_all_paired_point_estimates_positive':bool((all_source.paired_c0_minus_c2_delta_rr>0).all()),
             'all_source_all_paired_intervals_positive':bool((all_source.paired_c0_minus_c2_ci_low>0).all()),
             'all_source_all_C2_point_estimates_nonpositive':bool((all_source.c2_delta_rr<=0).all()),
             'source_score_hashes':hashes,
             'uncertainty':'User-cluster percentile bootstrap within each fixed mapping. 97.5% intervals retain the two-scope convention; no multiplicity control across mappings/models or independent-confirmation claim. Mapping repeats are not independent data.',
             'claim_ceiling':'Sensitivity to five predeclared cyclic-offset donor maps at the three existing UniSAR checkpoints, not arbitrary matching algorithms or new architectures.'}
    table.to_csv(out/'donor_mapping_sensitivity.csv',index=False)
    (out/'donor_mapping_sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
    compact=all_source[['mapping_seed','checkpoint','c2_delta_rr','c2_ci_low','c2_ci_high','paired_c0_minus_c2_delta_rr','paired_c0_minus_c2_ci_low','paired_c0_minus_c2_ci_high']].copy()
    for col in compact.columns[2:]:compact[col]*=1000
    (out/'RESULTS.md').write_text('# R266 donor-mapping sensitivity\n\nAll-source values in 10^-3 RR, intervals 97.5%.\n\n'+'```text\n'+compact.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```'+'\n\n'+summary['uncertainty']+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','source_score_hashes']},indent=2))


if __name__=='__main__':main()
