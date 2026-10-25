#!/usr/bin/env python3
"""Append the ten independently checked fixed precision fills to the holder.

This step preserves every previously declared holder mesh and the immutable
V5 source.  The new pieces are permanent, have no cuts or removable bodies,
and are included in both maximum retained stock and the fixed micro-fill
aggregate.  The complete release build/removal checks remain independent.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import manifold3d as mf

import custom_holder_supports as H

ROOT=H.ROOT
OUT=H.OUT
CANDIDATES=ROOT/'reports/fixed_precision_repair_candidates.json'
SCOPE=ROOT/'reports/fixed_precision_repair_scope_checks.json'
REPORT=ROOT/'reports/fixed_precision_repair_integration.json'
STAGE=OUT/'fixed_precision_integration'
EXPECTED_COUNT=1274
EXPECTED_FULL_SHA='11983193cc5017265e636691cbb5b1dfe015f4beba8b9a9ad4c96fdb9a76f290'
EXPECTED_CANDIDATE_SHA='3cf3abc9e31032df0eda238e33b20c986263b5eaf90ea3986e9d1090f1ba1275'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();started=time.monotonic()
    paths=[OUT/'holder_supports.json',ROOT/'reports/custom_holder_supports.json',OUT/'generation_checkpoint.json']
    documents=[json.loads(p.read_text()) for p in paths]
    rows=documents[0]['parts']
    if len(rows)==EXPECTED_COUNT+10 and all(r.get('precision_fixed_fill') for r in rows[EXPECTED_COUNT:]):
        print('The ten declared fixed fills are already integrated.',flush=True)
        return
    require(len(rows)==EXPECTED_COUNT,'Unexpected holder part count')
    require(all(d['parts']==rows for d in documents),'Holder manifests disagree')
    require(H.sha(OUT/'holder_supports.npz')==EXPECTED_FULL_SHA,'Unexpected holder aggregate')
    require(H.sha(CANDIDATES)==EXPECTED_CANDIDATE_SHA,'Unexpected fixed repair manifest')
    candidates=json.loads(CANDIDATES.read_text());scope=json.loads(SCOPE.read_text())
    require(candidates['local_geometry_passed'] and candidates['repair_count']==10,'Local fixed-fill checks failed')
    require(scope.get('passed') is True and len(scope['repairs'])==10,'Independent scope checks failed')
    # The checker has validated these exact original, support and mechanism
    # inputs.  Verify them again before making an integration commit.
    stale=[p for p,d in scope['input_sha256'].items() if H.sha(ROOT/p)!=d]
    require(not stale,'Independent fixed-fill scope inputs changed: '+str(stale))
    scope_by_id={r['id']:r for r in scope['repairs']}
    require(len(scope_by_id)==10,'Duplicate independent scope record')
    protected=[ROOT/'cache/original_v5.npz',*paths,
        OUT/'holder_supports.npz',OUT/'holder_retained_stumps.npz',OUT/'holder_micro_reinforcements.npz',
        CANDIDATES,SCOPE]
    protected_hashes={str(p.relative_to(ROOT)):H.sha(p) for p in protected}
    prior_mesh_hashes={name:H.sha(OUT/name) for r in rows for name in
        [r['full_mesh_file'],r['retained_mesh_file'],r['removable_mesh_file']]}
    STAGE.mkdir(exist_ok=True);before=STAGE/'before';before.mkdir(exist_ok=True)
    for p in paths:
        target=before/('reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name)
        shutil.copy2(p,target)
    new_rows=[];new_solids=[];generated=[]
    for offset,candidate in enumerate(candidates['repairs']):
        index=EXPECTED_COUNT+offset;check=scope_by_id[candidate['id']]
        require(check['passed'] is True and check['persistent_fixed_contact']['passed'] is True,
            'Failed contact/exclusion check for '+candidate['id'])
        require(check['mesh_sha256']==candidate['mesh_sha256'],'Scope/candidate mesh mismatch')
        source=ROOT/candidate['mesh_file']
        require(H.sha(source)==candidate['mesh_sha256'],'Changed candidate mesh')
        full,_,_=H.load(source)
        require(not full.is_empty() and H.volume(full)>0,'Empty fixed repair')
        require(candidate['local_after_strict_probe']['passed'] is True,'Strict local repair failed')
        names={k:f'holder_support_{index:04d}_{suffix}.npz' for k,suffix in
            [('full_mesh_file','full'),('retained_mesh_file','retained'),('removable_mesh_file','removable')]}
        for key in ('full_mesh_file','retained_mesh_file'):
            shutil.copy2(source,STAGE/names[key]);generated.append(names[key])
        H.save(STAGE/names['removable_mesh_file'],mf.Manifold());generated.append(names['removable_mesh_file'])
        row=dict(index=index,kind='fixed_micro_reinforcement',temporary=False,
            root_mm=candidate['root_mm'],target_mm=candidate['target_mm'],
            bounds_mm=candidate['bounds_mm'],volume_mm3=candidate['volume_mm3'],
            actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'],
            root_contact_volume_mm3=check['persistent_fixed_contact']['contact_volume_mm3'],
            moving_envelope_clearance_mm=check['categories']['continuous_motion_envelope']['clearance_lower_bound_mm'],
            neck_count=0,removal='Permanent fixed-holder precision fill; do not cut.',
            generation_source_sha256=candidates['source_sha256'],
            precision_fixed_fill=dict(id=candidate['id'],purpose=candidate['repair_purpose'],
                candidate_manifest=str(CANDIDATES.relative_to(ROOT)),candidate_manifest_sha256=H.sha(CANDIDATES),
                source_mesh_file=candidate['mesh_file'],source_mesh_sha256=candidate['mesh_sha256'],
                pre_integration_scope_report=str(SCOPE.relative_to(ROOT)),
                pre_integration_scope_report_sha256=H.sha(SCOPE),
                persistent_fixed_contact=check['persistent_fixed_contact'],
                local_after_strict_probe=candidate['local_after_strict_probe'],
                occupied_context_definition=candidate['occupied_context_definition'],
                global_build_and_external_removal_required=True),
            passed=True,**names)
        new_rows.append(row);new_solids.append(full)
    final_rows=copy.deepcopy(rows)+new_rows
    full_parts=[H.load(OUT/r['full_mesh_file'])[0] for r in rows]+new_solids
    retained_parts=[H.load(OUT/r['retained_mesh_file'])[0] for r in rows]+new_solids
    fixed_parts=[p for p,r in zip(full_parts,final_rows) if r['kind']=='fixed_micro_reinforcement']
    print('Rebuilding the three holder aggregates from all 1,284 declared pieces.',flush=True)
    aggregates={'holder_supports.npz':H.union(full_parts),
        'holder_retained_stumps.npz':H.union(retained_parts),
        'holder_micro_reinforcements.npz':H.union(fixed_parts)}
    for name,solid in aggregates.items():H.save(STAGE/name,solid)
    new_hashes={name:H.sha(STAGE/name) for name in aggregates}
    addition_total=sum(c['actual_added_volume_over_occupied_context_mm3'] for c in candidates['repairs'])
    proof=dict(passed=True,applied=args.apply,part_count_before=len(rows),part_count_after=len(final_rows),
        added_indices=list(range(EXPECTED_COUNT,EXPECTED_COUNT+10)),added_kind='fixed_micro_reinforcement',
        permanent_added_part_count=10,temporary_part_count_unchanged=True,
        all_previous_declared_meshes_unchanged=True,immutable_original_unchanged=True,
        existing_declared_mesh_sha256=prior_mesh_hashes,pre_integration_input_sha256=protected_hashes,
        candidate_manifest=str(CANDIDATES.relative_to(ROOT)),candidate_manifest_sha256=H.sha(CANDIDATES),
        pre_integration_scope_report=str(SCOPE.relative_to(ROOT)),pre_integration_scope_report_sha256=H.sha(SCOPE),
        prior_full_aggregate_sha256=EXPECTED_FULL_SHA,aggregate_sha256=new_hashes,
        whole_new_piece_volume_mm3=sum(c['volume_mm3'] for c in candidates['repairs']),
        actual_added_volume_over_occupied_context_mm3=addition_total,
        original_signed_context_difference_not_used_as_added_volume=True,
        scope_refresh_required_after_integration=True,final_whole_model_gates_required=True,
        integration_source_sha256=H.sha(__file__),elapsed_seconds=time.monotonic()-started)
    addition=dict(kind='explicit_fixed_precision_fills',added_part_indices=proof['added_indices'],
        report=str(REPORT.relative_to(ROOT)),passed=True,
        actual_added_volume_over_occupied_context_mm3=addition_total,
        source_sha256=H.sha(__file__))
    for document,path in zip(documents,paths):
        document['parts']=copy.deepcopy(final_rows)
        document.setdefault('post_generation_repairs',[]).append(addition)
        if path.name!='generation_checkpoint.json':
            document.update(supports=len(final_rows),
                permanent_micro_reinforcement_count=sum(r['kind']=='fixed_micro_reinforcement' for r in final_rows),
                permanent_micro_reinforcement_volume_mm3=H.volume(aggregates['holder_micro_reinforcements.npz']),
                support_volume_mm3=H.volume(aggregates['holder_supports.npz']),
                generation_checks_passed=all(r['passed'] for r in final_rows),passed=False)
        name='reports_custom_holder_supports.json' if path.parent.name=='reports' else path.name
        (STAGE/name).write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    # Commit only if the exact protected input snapshot is still unchanged.
    require(all(H.sha(ROOT/p)==d for p,d in protected_hashes.items()),'Protected integration input changed')
    require(all(H.sha(OUT/p)==d for p,d in prior_mesh_hashes.items()),'A prior holder mesh changed')
    if args.apply:
        for name in generated+list(aggregates):(STAGE/name).replace(OUT/name)
        for path in paths:
            name='reports_custom_holder_supports.json' if path.parent.name=='reports' else path.name
            (STAGE/name).replace(path)
        require(all(H.sha(OUT/p)==d for p,d in prior_mesh_hashes.items()),'Prior holder mesh changed during commit')
        require(H.sha(ROOT/'cache/original_v5.npz')==protected_hashes['cache/original_v5.npz'],'V5 source changed')
        require(all(H.sha(OUT/name)==digest for name,digest in new_hashes.items()),'Committed aggregate hash mismatch')
    REPORT.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ['existing_declared_mesh_sha256','pre_integration_input_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
