#!/usr/bin/env python3
"""Audit saved C1 construction and preflight C2 mappings; no model inference.

The original deterministic map functions are loaded directly from their AST.
Only their tensor container is replaced by an int64 NumPy container, allowing
metadata analysis without installing torch. Original map hashes must match.
Native history sampling is executed from the frozen official Sampler class.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class IntArray(np.ndarray):
    def numpy(self):
        return np.asarray(self)


CONTAINERS = SimpleNamespace(
    from_numpy=lambda x: np.asarray(x).view(IntArray),
    zeros=lambda n, dtype: np.zeros(n, dtype=dtype).view(IntArray),
    long=np.int64,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_defs(path: Path, names: list[str], env: dict) -> None:
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'missing definitions in {path}')
    source = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes)
    exec(compile(source, str(path), 'exec'), env)


def retention(ids: np.ndarray, overlaps: np.ndarray, known: np.ndarray, mask: np.ndarray) -> dict:
    sub = ids[mask]
    active = sub > 0
    nslots = int(active.sum())
    yes = overlaps[sub] & active
    have = known[sub] & active
    per_row = yes.sum(axis=1)
    counts = active.sum(axis=1)
    return {'rows': len(sub), 'source_occurrences': nslots,
            'overlap_occurrences': int(yes.sum()),
            'overlap_occurrence_fraction': float(yes.sum()/nslots),
            'both_sides_have_category_fraction': float(have.sum()/nslots),
            'overlap_given_both_known_fraction': float(yes.sum()/have.sum()) if have.any() else None,
            'all_slots_retain_category_rows': int((per_row == counts).sum()),
            'all_slots_retain_category_fraction': float((per_row == counts).mean()),
            'at_least_one_retains_category_fraction': float((per_row > 0).mean()),
            'no_slot_retains_category_fraction': float((per_row == 0).mean()),
            'mean_row_retained_slot_fraction': float((per_row/counts).mean())}


def history_retention(ids: np.ndarray, mapping: np.ndarray, categories: list) -> dict:
    """Compare whole-history category unions and per-session category prevalence."""
    category_ids=sorted(set().union(*categories))
    if not category_ids or max(category_ids)>=64:
        raise ValueError('history category-bitset contract changed')
    bits=np.zeros(len(categories)+1,dtype=np.uint64)
    for sid,cats in enumerate(categories):
        for cat in cats:bits[sid] |= np.uint64(1)<<np.uint64(cat)
    original=np.bitwise_or.reduce(bits[ids],axis=1)
    replacement=np.bitwise_or.reduce(bits[mapping[ids]],axis=1)
    count_bits=lambda x:np.fromiter((int(v).bit_count() for v in x),dtype=np.int16,count=len(x))
    na,nb=count_bits(original),count_bits(replacement)
    ni,nu=count_bits(original&replacement),count_bits(original|replacement)
    valid=(na>0)&(nb>0)
    a=np.empty((len(ids),len(category_ids)),dtype=np.float64);b=np.empty_like(a)
    for column,cat in enumerate(category_ids):
        lookup=(bits & (np.uint64(1)<<np.uint64(cat)))>0
        a[:,column]=lookup[ids].sum(axis=1)
        b[:,column]=lookup[mapping[ids]].sum(axis=1)
    a/=np.maximum(a.sum(axis=1,keepdims=True),1)
    b/=np.maximum(b.sum(axis=1,keepdims=True),1)
    hist_overlap=np.minimum(a,b).sum(axis=1)
    return {'rows':len(ids),'rows_with_categories_on_both_sides':int(valid.sum()),
            'mean_original_category_union_size':float(na[valid].mean()),
            'mean_replacement_category_union_size':float(nb[valid].mean()),
            'any_history_category_overlap_fraction':float((ni[valid]>0).mean()),
            'all_original_categories_still_present_fraction':float((ni[valid]==na[valid]).mean()),
            'exact_history_category_union_match_fraction':float((original[valid]==replacement[valid]).mean()),
            'mean_original_categories_retained_fraction':float((ni[valid]/na[valid]).mean()),
            'mean_history_category_jaccard':float((ni[valid]/nu[valid]).mean()),
            'mean_category_prevalence_histogram_overlap':float(hist_overlap[valid].mean()),
            'definition':'Unions aggregate all replaced source sessions; prevalence counts each category at most once per session and normalizes within the history. Rows with an empty category union on either side are excluded from these fractions.'}


def self_test() -> None:
    ids = np.array([[1,2,0],[2,3,0],[1,0,0]])
    overlap = np.array([False,True,False,True])
    d = retention(ids, overlap, np.array([False,True,True,True]), np.ones(3, bool))
    assert d['overlap_occurrences'] == 3 and d['source_occurrences'] == 5
    assert d['all_slots_retain_category_rows'] == 1
    assert np.isclose(d['mean_row_retained_slot_fraction'], 2/3)
    # Container conversion is exact int64 and changes no map values.
    a = np.arange(9,dtype=np.int64)
    assert CONTAINERS.from_numpy(a).numpy().tobytes() == a.tobytes()
    cats=[set(),{1},{2},{1},{3}]
    h=history_retention(np.array([[1,2],[1,0]]),np.array([0,2,1,4,3]),cats)
    # First history has every slot's category changed, but its category union is identical.
    assert h['exact_history_category_union_match_fraction']==.5
    assert h['mean_original_categories_retained_fraction']==.5
    print('C1 retention metadata self-test passed')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-data-root',type=Path)
    parser.add_argument('--official-repo',type=Path)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--original-c2-integrity',type=Path)
    parser.add_argument('--helper-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not all([args.processed_data_root,args.official_repo,args.baseline_root,args.original_c2_integrity,args.output_dir]):
        parser.error('data, official code, baseline, C2 reference and output paths are required')
    helpers=args.helper_root.resolve()
    processed=args.processed_data_root.resolve()
    runtime=processed.parent
    official=args.official_repo.resolve()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    priors=[json.loads((args.baseline_root/f'r233_unisar_seed_{s}_integrity.json').read_text()) for s in (20230601,20230602,20230603)]
    expected=priors[0]
    input_hashes={}
    for old,digest in expected['execution_lineage']['processed_input_sha256'].items():
        path=processed/('dataset' if Path(old).name=='rec_test.pkl' else 'vocab')/Path(old).name
        actual=sha(path)
        if actual != digest:raise ValueError(f'processed hash mismatch: {path.name}')
        input_hashes[str(path.relative_to(runtime))]=actual
    sampler=official/'utils/sampler.py'
    official_hashes={}
    for old,digest in expected['execution_lineage']['official_code_sha256'].items():
        rel=Path(old).parts[-2:]
        path=official.joinpath(*rel)
        actual=sha(path)
        if actual!=digest:raise ValueError(f'official code mismatch: {path.name}')
        official_hashes['/'.join(rel)]=actual
    env={'np':np,'pd':pd,'os':os,'Counter':Counter,'defaultdict':defaultdict,'hashlib':hashlib,
         'json':json,'torch':CONTAINERS,'Any':Any}
    r233=helpers/'unisar_source_scope_localization.py'
    r242=helpers/'unisar_semantic_control_ladder.py'
    load_defs(r233,['bucket_len','occurrence_bucket','nonzero_count','stable_slot_key','sha256_json','build_valid_source_derangement'],env)
    load_defs(r242,['item_count_bucket','frequency_bucket','source_category_sets','build_category_donor_map'],env)
    load_defs(official/'utils/const.py',['init_setting_KuaiSAR'],env)
    env['init_setting_KuaiSAR']()
    env['const']=SimpleNamespace(max_rec_his_len=env['max_rec_his_len'],max_src_session_his_len=env['max_src_session_his_len'])
    load_defs(sampler,['Sampler'],env)
    with (processed/'vocab/user_vocab.pkl').open('rb') as f:user_vocab=pickle.load(f)
    native=env['Sampler'](str(processed/'dataset/rec_test.pkl'),False,user_vocab)
    frame=native.data
    if len(frame)!=157192:raise ValueError('test split changed')
    ids=np.zeros((len(frame),env['max_src_session_his_len']),np.int64)
    for i,(user,rec_count,src_count) in enumerate(frame[['user_id','rec_his','src_session_his']].itertuples(index=False,name=None)):
        feed=native.get_all_his(int(user),int(rec_count),int(src_count))
        source=[int(sid) for sid,typ in zip(feed['all_his'][0],feed['all_his_type'][0]) if typ==2 and sid!=0]
        if not source:raise ValueError(f'empty source row {i}')
        ids[i,:len(source)]=source
    counts=Counter(map(int,ids[ids>0]))
    support_hash=env['sha256_json']([[sid,counts[sid]] for sid in sorted(counts)])
    if support_hash!=expected['donor_map']['observed_support_sha256']:raise ValueError('observed support hash mismatch')
    old_csv=pd.read_csv(args.baseline_root/'r233_unisar_seed_20230601_scope_scores.csv',usecols=['official_row_index','user_id','source_slot_count'])
    assert np.array_equal(old_csv.official_row_index.to_numpy(),np.arange(len(frame)))
    assert np.array_equal(old_csv.user_id.to_numpy(),frame.user_id.to_numpy())
    assert np.array_equal(old_csv.source_slot_count.to_numpy(),(ids>0).sum(axis=1))
    with (processed/'vocab/src_session_vocab_np.pkl').open('rb') as f:src_map=pickle.load(f)
    with (processed/'vocab/item_vocab_np.pkl').open('rb') as f:item_map=pickle.load(f)
    categories=env['source_category_sets'](src_map,np.asarray(item_map['first_level_category_id']))
    neutral=len(categories)
    c1,c1_summary=env['build_valid_source_derangement'](src_map,counts,neutral,int(expected['intervention_seed']))
    for prior in priors:
        if c1_summary!=prior['donor_map']:raise ValueError('C1 reconstruction differs from saved summary')
    canonical=json.loads(args.original_c2_integrity.read_text())
    c2,c2_summary=env['build_category_donor_map'](src_map,np.asarray(item_map['first_level_category_id']),counts,neutral,20260712)
    if c2_summary!=canonical['category_donor_map']:raise ValueError('C2 reconstruction differs from saved summary')
    overlap=np.zeros(neutral+1,bool);known=np.zeros(neutral+1,bool);exact=np.zeros(neutral+1,bool)
    for sid in counts:
        src,dst=categories[sid],categories[int(c1[sid])]
        overlap[sid]=bool(src&dst);known[sid]=bool(src and dst);exact[sid]=bool(src and src==dst)
    latest=np.array([row[n-1] for row,n in zip(ids,(ids>0).sum(axis=1))],dtype=np.int64).reshape(-1,1)
    latest_ok=c2[latest[:,0]]>0
    all_ok=((c2[ids]>0)|(ids==0)).all(axis=1)
    if (int(latest_ok.sum()),int(all_ok.sum()))!=(87044,66139):raise ValueError('C2 supports changed')
    scopes={
        'latest_full_test':retention(latest,overlap,known,np.ones(len(frame),bool)),
        'latest_C2_support':retention(latest,overlap,known,latest_ok),
        'all_full_test':retention(ids,overlap,known,np.ones(len(frame),bool)),
        'all_C2_common_support':retention(ids,overlap,known,all_ok),
    }
    history_scopes={
        'all_full_test':history_retention(ids,c1,categories),
        'all_C2_common_support':history_retention(ids[all_ok],c1,categories),
    }
    id_list=np.array(sorted(counts),dtype=np.int64)
    weights=np.array([counts[int(sid)] for sid in id_list])
    source_summary={'unique_source_ids':len(id_list),'category_overlap_fraction':float(overlap[id_list].mean()),
                    'exact_nonempty_category_set_fraction':float(exact[id_list].mean()),
                    'slot_weighted_exact_category_set_fraction':float(np.average(exact[id_list],weights=weights)),
                    'slot_weighted_category_overlap_fraction':float(np.average(overlap[id_list],weights=weights)),
                    'C1_route_counts':c1_summary['route_counts'],
                    'strict_profile_source_id_fraction':c1_summary['route_counts']['item_query_and_frequency_bucket']/len(id_list)}
    rows=pd.DataFrame({'official_row_index':np.arange(len(frame)), 'source_slot_count':(ids>0).sum(axis=1),
                       'latest_C1_category_overlap':overlap[latest[:,0]],'latest_C2_eligible':latest_ok,
                       'all_C2_eligible':all_ok,'C1_overlap_slot_count':overlap[ids].sum(axis=1)})
    # No row-level user identifiers or source IDs are exported.
    rows.to_csv(out/'c1_category_retention_rows.csv',index=False)
    result={'schema':'focalswap.r266.c1_category_retention.v1','status':'COMPLETE','new_model_inference':False,
            'definition':'At least one shared positive first-level item category between original search session and its original C1 donor.',
            'source_summary':source_summary,'scopes':scopes,'whole_history_categories':history_scopes,'original_C1':c1_summary,
            'verification':{'processed_hashes':input_hashes,'registered_official_code_hashes':official_hashes,
                            'native_sampler_snapshot_sha256':sha(sampler),
                            'official_git_metadata_checked':False,
                            'native_row_identity_and_source_counts':True,'C1_all_three_seed_summaries_exact_match':True,
                            'original_C2_summary_exact_match':True,'original_C2_supports_match':True,
                            'metadata_only_numpy_container':True,
                            'git_metadata_note':'This metadata-only audit requires exact registered input/code hashes, support, source counts, C1/C2 map hashes and row masks; it does not inspect Git metadata.'},
            'code_hashes':{p.name:sha(p) for p in [r233,r242,Path(__file__).resolve()]},
            'claim_ceiling':'Describes actual first-level category preservation. No semantic-equivalence or unique-mechanism conclusion.'}
    (out/'c1_category_retention.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_summary':source_summary,'scopes':scopes},indent=2),flush=True)
    # Preflight all five predeclared C2 mappings without scoring any checkpoint.
    maps={20260712:c2};mapping_records=[]
    valid=c2>0
    for seed in (20260921,20260922,20260923,20260924,20260925):
        mapping,summary=env['build_category_donor_map'](src_map,np.asarray(item_map['first_level_category_id']),counts,neutral,seed)
        if not np.array_equal(mapping>0,valid):raise ValueError('mapping changes source support')
        if summary['induced_occurrence_total_variation_on_eligible_support']>0.15:raise ValueError('mapping TV exceeds the registered gate')
        changed=(mapping!=c2)&valid
        mapped_ids=id_list[valid[id_list]]
        changed_occ=sum(counts[int(sid)] for sid in mapped_ids if changed[sid])
        record={'mapping_seed':seed,'donor_map_sha256':summary['donor_map_sha256'],
                'eligible_source_ids':summary['eligible_source_ids'],
                'source_ids_changed_from_original':int(changed.sum()),
                'source_id_changed_fraction':float(changed[valid].mean()),
                'occurrence_changed_fraction':changed_occ/summary['eligible_source_slot_occurrences'],
                'latest_eligible_rows':int(latest_ok.sum()),'all_eligible_rows':int(all_ok.sum()),
                'latest_donor_changed_fraction':float(changed[latest[latest_ok,0]].mean()),
                'all_rows_any_donor_changed_fraction':float(changed[ids[all_ok]].any(axis=1).mean()),
                'induced_occurrence_tv':summary['induced_occurrence_total_variation_on_eligible_support']}
        maps[seed]=mapping;mapping_records.append(record)
        print(f'C2 mapping {seed}: metadata validity and support pass',flush=True)
    immutable=np.logical_and.reduce([m==c2 for seed,m in maps.items() if seed!=20260712])&valid
    pairwise=[]
    for a in maps:
        for b in maps:
            if a>=b:continue
            pairwise.append({'mapping_a':a,'mapping_b':b,'different_source_id_fraction':float((maps[a][valid]!=maps[b][valid]).mean())})
    meta={'schema':'focalswap.r266.c2_mapping_preflight.v1','status':'PASS_METADATA_ONLY',
          'new_model_inference':False,'checkpoint_results_available':False,
          'mapping_family':'sorted-stratum cyclic-offset derangements; not arbitrary matching algorithms',
          'original_map_sha256':c2_summary['donor_map_sha256'],'original_C1_map_sha256':c1_summary['donor_map_sha256'],
          'mapping_records':mapping_records,'pairwise_mapping_differences':pairwise,
          'unchanged_under_all_five_source_ids':int(immutable.sum()),
          'unchanged_under_all_five_source_id_fraction':float(immutable[valid].mean()),
          'note':'Actual checkpoint contribution/CI matrix remains unexecuted until the Fudan backend is reachable.'}
    (out/'c2_mapping_metadata_preflight.json').write_text(json.dumps(meta,indent=2)+'\n')
    pd.DataFrame(mapping_records).to_csv(out/'c2_mapping_metadata_preflight.csv',index=False)


if __name__=='__main__':
    main()
