#!/usr/bin/env python3
"""Continue a frozen external-removal proof without replaying its old moves.

The prior proof, every prior source byte and every saved extracted fragment
and body kerf are verified. The exact remaining scene is recreated, then only
the remaining operations are planned with a separately hash-bound overlay. No printed geometry is modified.
"""
from pathlib import Path
import argparse,copy,hashlib,io,json,math,time,zipfile
import external_removal_extended_v6 as S
import plan_external_removal_v6 as P

def require(condition,message):
    if not condition:raise RuntimeError(message)

def checked_output(entry,path_key):
    path=P.ROOT/entry[path_key];data=P.frozen_bytes(path)
    require(hashlib.sha256(data).hexdigest()==entry['sha256'],'Prior output changed: '+str(path))
    with zipfile.ZipFile(io.BytesIO(data)) as z:require(z.testzip() is None,'Prior output CRC failed')

def small_nonnegative(value):
    return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=P.TOL

def verify_prior(prior):
    require(prior['numerical_positive_intersection_tolerance_mm3']==P.TOL,'Different prior collision tolerance')
    require(prior['assembly_geometry_binding'].get('passed') is True,'Prior geometry binding failed')
    inputs=prior['input_files'];require(len({r['path'] for r in inputs})==len(inputs),'Duplicate prior input path')
    for row in inputs:
        P.frozen_bytes(P.ROOT/row['path'])
        require(P.INPUT_HASHES[row['path']]==row['sha256'],'Prior source changed: '+row['path'])
    require(any(r['path']=='source/plan_external_removal_v6.py' for r in inputs),'Prior planner source is not bound')
    require(any(r['path']=='source/external_removal_sequential_v6.py' for r in inputs),'Prior sequential verifier source is not bound')
    plans=copy.deepcopy(prior['plans'])
    require(prior['planned_support_count']==len(plans),'Prior declared plan count differs from actual plans')
    require(len({r['id'] for r in plans})==len(plans),'Duplicate prior sequence ID')
    for sequence,row in enumerate(plans):
        require(row.get('passed') is True and row['sequence_index']==sequence,'Invalid prior sequence')
        if row['group']=='sphere':require(row['endpoint_cut_validation']['passed'] is True,'Prior endpoint cuts failed')
        if row['mode']=='explicit_body_cuts':
            checks=row.get('body_cut_partition_checks')
            require(checks is not None and all(small_nonnegative(x) for x in checks.values()),'Prior cut coverage failed')
            require(small_nonnegative(row['kerf_complete_obstacle_state_overlap_mm3']),'Prior body kerf collision')
            require(row.get('actual_body_kerf') is not None,'Prior actual body kerf missing')
            checked_output(row['actual_body_kerf'],'path')
        if row['mode']=='endpoint_cuts_only':require(not row['fragments'],'Cut-only step has an extracted body')
        for k,fragment in enumerate(row['fragments']):
            checked_output(fragment,'mesh_file')
            require(fragment.get('exits_initial_scene_below_or_above') is True,'Prior fragment did not exit')
            require(all(small_nonnegative(leg['maximum_swept_intersection_mm3']) for leg in fragment['translation_legs']),'Prior swept collision')
            bb=fragment['terminal_bounds_mm'];lo,hi=prior['initial_scene_z_bounds_mm']
            escaped=bb[5]<lo if row['kind']=='plate_perimeter_tack' else bb[2]>hi
            require(escaped,'Prior saved terminal bounds do not leave the scene')
            # The original verifier emitted the ordered nonempty partition
            # fragments in that same list order. New sequential proofs record
            # their original partition indices explicitly.
            fragment.setdefault('body_section_index',k)
        order=[f['body_section_index'] for f in row['fragments']]
        require(sorted(order)==list(range(len(order))),'Invalid prior partition correspondence')
        row.setdefault('fragment_extraction_order',order)
    return plans

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--prior-sha256',required=True)
    ap.add_argument('--candidates',type=Path)
    a=ap.parse_args();started=time.monotonic();target=P.ROOT/'reports/v6_external_removal_validation.json'
    prior_path=a.prior.resolve();require(prior_path!=target,'Prior proof must be an immutable separate snapshot')
    require(prior_path.is_relative_to(P.ROOT),'Prior proof outside project')
    P.frozen_bytes(__file__);prior=P.metadata(prior_path)
    require(P.INPUT_HASHES[str(prior_path.relative_to(P.ROOT))]==a.prior_sha256,'Wrong prior proof bytes')
    plans=verify_prior(prior);prior_count=len(plans)
    # The old proof is completely verified before the new overlay is installed.
    S.install()
    if a.candidates:
        candidate_path=a.candidates.resolve();candidates=P.metadata(candidate_path)
        P.ROUTE_SEEDS={r['id']:r for r in candidates['plans']}
        require(len(P.ROUTE_SEEDS)==len(candidates['plans']),'Duplicate candidate instruction IDs')
    items,permanent,_,all_full=P.load_parts();del all_full
    require(len(items)==prior['initial_support_count'],'Initial source count changed')
    by_id={r['id']:r for r in items};require(len(by_id)==len(items),'Duplicate source operation ID')
    removed={r['id'] for r in plans};require(removed<=set(by_id),'Prior operation missing from current sources')
    for row in plans:
        item=by_id[row['id']]
        require(row['group']==item['group'] and row['source_index']==item['source_index'] and row['kind']==item['kind'],'Prior/current operation identity differs')
    expected_remaining=set(by_id)-removed
    require(len({r['id'] for r in prior['remaining_failures']})==len(prior['remaining_failures']),'Duplicate prior unfinished ID')
    require({r['id'] for r in prior['remaining_failures']}==expected_remaining,'Prior unfinished set is not explicit and complete')
    original_path=P.ROOT/'cache/original_v5.npz';digest=P.sha(original_path)
    require(digest==prior['original_v5_sha256'],'Original model changed')
    surface=P.H.Surface(original_path);P.H.ACTIVE_SURFACE=surface;original=surface.solid
    require(P.sha(original_path)==digest,'Original model changed while loading')
    P.SCENE_MIN_Z,P.SCENE_MAX_Z=prior['initial_scene_z_bounds_mm']
    field=P.ObstacleField(items,permanent);field.active.difference_update(removed)
    require(field.active==expected_remaining,'Incorrect recreated remaining scene')
    seed_order={key:i for i,key in enumerate(P.ROUTE_SEEDS)}
    remaining=sorted((r for r in items if r['id'] in field.active),key=lambda r:
        (0,seed_order[r['id']],0,r['id']) if r['id'] in seed_order else
        (1,0 if r['group']=='holder' else 1,-r['source_index'],r['id']))
    print('Verified immutable prior proof:',prior_count,'accepted;',len(remaining),'remain.',flush=True)
    failures=[];broad_search=False
    while remaining:
        progress=False;failures=[]
        for item in list(remaining):
            plan,error=P.make_plan(item,original,field.context(item['id']),broad_search=broad_search)
            if plan is None:
                failures.append(dict(id=item['id'],**error));print('Deferred',item['id'],error['reason'],flush=True);continue
            row=P.save_plan(item,plan,len(plans));plans.append(row)
            remaining=[other for other in remaining if other is not item];field.active.remove(item['id']);progress=True
            print('Removal',len(plans),'/',len(items),item['id'],row['mode'],flush=True)
            checkpoint=dict(complete=False,initial_support_count=len(items),plans=plans,
                immutable_prior_proof=str(prior_path.relative_to(P.ROOT)),prior_proof_sha256=a.prior_sha256,
                remaining_ids=[r['id'] for r in remaining])
            (P.OUT/'extended_continuation_checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
        if not progress:
            if not broad_search:broad_search=True;print('Searching all remaining sequential whole/segmented routes.',flush=True);continue
            break
    changed=[path for path,digest in P.INPUT_HASHES.items() if P.sha(P.ROOT/path)!=digest]
    require(not changed,'Continuation input changed: '+str(changed))
    report=copy.deepcopy(prior)
    report.update(passed=not remaining,plans=plans,planned_support_count=len(plans),remaining_failures=failures,
        input_files=[dict(path=path,sha256=digest) for path,digest in sorted(P.INPUT_HASHES.items())],
        continuation_proof=dict(immutable_prior_report=str(prior_path.relative_to(P.ROOT)),prior_sha256=a.prior_sha256,
            reused_prior_operation_count=prior_count,all_prior_input_bytes_unchanged=True,
            all_prior_saved_fragment_and_kerf_bytes_verified=True,prior_partition_indices_normalized_without_geometry_change=True,
            initial_recreated_active_ids=sorted(expected_remaining),source='source/continue_external_removal_extended_v6.py',
            all_prior_accepted_ids_disabled_only_after_their_previously_certified_exit=True),
        continuation_candidate_instructions=(str(a.candidates.resolve().relative_to(P.ROOT)) if a.candidates else None),
        candidate_routes_fully_revalidated_count=sum(bool(row.get('candidate_route_revalidated')) for row in plans),
        search_order='Immutable accepted prefix reused after full byte binding; remaining candidates, preferred sequential departures, then full sequential direction/body-cut search.',
        continuation_newly_planned_count=len(plans)-prior_count,
        elapsed_seconds=prior.get('elapsed_seconds',0.)+time.monotonic()-started,
        continuation_elapsed_seconds=time.monotonic()-started)
    temporary=target.with_suffix('.json.part');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(target)
    S.finish_report(target)
    print('Continuation complete:',len(plans),'/',len(items),'accepted;',len(remaining),'remaining.',flush=True)
    if remaining:raise SystemExit(2)

if __name__=='__main__':main()
