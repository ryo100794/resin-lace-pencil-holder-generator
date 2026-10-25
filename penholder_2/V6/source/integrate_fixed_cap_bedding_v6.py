#!/usr/bin/env python3
"""Append independently proved fixed cap bedding to the frozen V6 holder.

Every existing printed, retained and removable mesh remains byte-identical.
The explicit new pieces are permanent and receive empty removable meshes.
"""
from pathlib import Path
import argparse,copy,json,shutil,time
import manifold3d as mf
import custom_holder_supports as H

ROOT=H.ROOT;OUT=H.OUT
REPORT=ROOT/'reports/fixed_precision_cap_bedding_integration.json'
STAGE=OUT/'cap_bedding_integration'
EXPECTED_COUNT=1285
EXPECTED_FULL='9322225d19a4cb86577f8e9e6417cbe39929fcdef6ec6c3b5fb5504555935921'

def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--scope',type=Path,required=True)
    ap.add_argument('--candidate-sha256',required=True)
    ap.add_argument('--apply',action='store_true');a=ap.parse_args();start=time.monotonic()
    candidate_path=a.candidates.resolve();scope_path=a.scope.resolve()
    require(candidate_path.is_relative_to(ROOT) and scope_path.is_relative_to(ROOT),'Evidence outside project')
    require(H.sha(candidate_path)==a.candidate_sha256,'Unexpected candidate manifest')
    candidates=json.loads(candidate_path.read_text());scope=json.loads(scope_path.read_text())
    count=len(candidates['repairs']);ids=[c['id'] for c in candidates['repairs']]
    require(candidates['local_geometry_passed'] is True and count>0 and len(set(ids))==count,'Local candidate gate failed')
    require(scope.get('passed') is True and len(scope['repairs'])==count,'Independent exclusion gate failed')
    checks={r['id']:r for r in scope['repairs']}
    require(set(checks)==set(ids),'Scope/candidate record mismatch')
    paths=[OUT/'holder_supports.json',ROOT/'reports/custom_holder_supports.json',OUT/'generation_checkpoint.json']
    documents=[json.loads(p.read_text()) for p in paths];rows=documents[0]['parts']
    if len(rows)==EXPECTED_COUNT+count and all(r.get('precision_fixed_fill',{}).get('candidate_manifest_sha256')==a.candidate_sha256 for r in rows[EXPECTED_COUNT:]):
        print('This exact cap-bedding set is already integrated.',flush=True);return
    require(len(rows)==EXPECTED_COUNT and all(d['parts']==rows for d in documents),'Unexpected holder manifests')
    require(H.sha(OUT/'holder_supports.npz')==EXPECTED_FULL,'Unexpected holder aggregate')
    stale=[p for p,digest in scope['input_sha256'].items() if H.sha(ROOT/p)!=digest]
    require(not stale,'Scope inputs changed: '+str(stale))
    protected=[ROOT/'cache/original_v5.npz',candidate_path,scope_path,*paths,
        OUT/'holder_supports.npz',OUT/'holder_retained_stumps.npz',OUT/'holder_micro_reinforcements.npz']
    hashes={str(p.relative_to(ROOT)):H.sha(p) for p in protected}
    old_mesh_hashes={name:H.sha(OUT/name) for r in rows for name in
        [r['full_mesh_file'],r['retained_mesh_file'],r['removable_mesh_file']]}
    STAGE.mkdir(exist_ok=True);before=STAGE/'before';before.mkdir(exist_ok=True)
    for p in paths:shutil.copy2(p,before/('reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name))
    new_rows=[];new_solids=[];generated=[]
    for offset,candidate in enumerate(candidates['repairs']):
        index=EXPECTED_COUNT+offset;check=checks[candidate['id']]
        require(check['passed'] is True and check['persistent_fixed_contact']['passed'] is True,'Unproved cap contact')
        require(candidate['local_after_strict_probe']['passed'] is True,'Strict local cap check failed')
        source=ROOT/candidate['mesh_file'];require(H.sha(source)==candidate['mesh_sha256']==check['mesh_sha256'],'Changed candidate mesh')
        full=H.load(source)[0];require(not full.is_empty() and H.volume(full)>0,'Empty fixed cap')
        names={k:f'holder_support_{index:04d}_{suffix}.npz' for k,suffix in
            [('full_mesh_file','full'),('retained_mesh_file','retained'),('removable_mesh_file','removable')]}
        require(not any((OUT/name).exists() for name in names.values()),'New index already has adopted geometry')
        for key in ['full_mesh_file','retained_mesh_file']:shutil.copy2(source,STAGE/names[key])
        H.save(STAGE/names['removable_mesh_file'],mf.Manifold());generated.extend(names.values())
        row=dict(index=index,kind='fixed_micro_reinforcement',temporary=False,
            root_mm=candidate['root_mm'],target_mm=candidate['target_mm'],bounds_mm=candidate['bounds_mm'],
            volume_mm3=candidate['volume_mm3'],actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'],
            root_contact_volume_mm3=check['persistent_fixed_contact']['contact_volume_mm3'],
            moving_envelope_clearance_mm=check['categories']['continuous_motion_envelope']['clearance_lower_bound_mm'],
            neck_count=0,removal='Permanent fixed-holder cap bedding; do not cut.',
            generation_source_sha256=candidates['source_sha256'],passed=True,
            precision_fixed_fill=dict(id=candidate['id'],purpose=candidate['repair_purpose'],
                candidate_manifest=str(candidate_path.relative_to(ROOT)),candidate_manifest_sha256=H.sha(candidate_path),
                source_mesh_file=candidate['mesh_file'],source_mesh_sha256=candidate['mesh_sha256'],
                pre_integration_scope_report=str(scope_path.relative_to(ROOT)),pre_integration_scope_report_sha256=H.sha(scope_path),
                persistent_fixed_contact=check['persistent_fixed_contact'],local_after_strict_probe=candidate['local_after_strict_probe'],
                source_evidence=copy.deepcopy(candidate),global_build_and_external_removal_required=True),**names)
        new_rows.append(row);new_solids.append(full)
    final_rows=copy.deepcopy(rows)+new_rows
    fulls=[H.load(OUT/r['full_mesh_file'])[0] for r in rows]+new_solids
    retained=[H.load(OUT/r['retained_mesh_file'])[0] for r in rows]+new_solids
    aggregates={'holder_supports.npz':H.union(fulls),'holder_retained_stumps.npz':H.union(retained),
        'holder_micro_reinforcements.npz':H.union([p for p,r in zip(fulls,final_rows) if r['kind']=='fixed_micro_reinforcement'])}
    for name,mesh in aggregates.items():H.save(STAGE/name,mesh)
    aggregate_hashes={name:H.sha(STAGE/name) for name in aggregates}
    total=sum(r['actual_added_volume_over_occupied_context_mm3'] for r in candidates['repairs'])
    proof=dict(passed=True,applied=a.apply,part_count_before=len(rows),part_count_after=len(final_rows),
        added_indices=list(range(EXPECTED_COUNT,EXPECTED_COUNT+count)),added_kind='fixed_micro_reinforcement',
        temporary_part_count_unchanged=True,all_previous_declared_meshes_unchanged=True,immutable_original_unchanged=True,
        pre_integration_input_sha256=hashes,existing_declared_mesh_sha256=old_mesh_hashes,aggregate_sha256=aggregate_hashes,
        candidate_manifest=str(candidate_path.relative_to(ROOT)),candidate_manifest_sha256=H.sha(candidate_path),
        actual_added_volume_over_occupied_context_mm3=total,added_volume_basis='Sum of independently evaluated per-pad occupied-set additions.',
        whole_new_piece_volume_mm3=sum(r['volume_mm3'] for r in candidates['repairs']),
        scope_refresh_required_after_integration=True,final_whole_model_gates_required=True,
        integration_source_sha256=H.sha(__file__),elapsed_seconds=time.monotonic()-start)
    addition=dict(kind='explicit_fixed_cap_bedding',added_part_indices=proof['added_indices'],report=str(REPORT.relative_to(ROOT)),passed=True,
        candidate_manifest=str(candidate_path.relative_to(ROOT)),scope_report=str(scope_path.relative_to(ROOT)),
        actual_added_volume_over_occupied_context_mm3=total,source_sha256=H.sha(__file__))
    for d,p in zip(documents,paths):
        d['parts']=copy.deepcopy(final_rows);d.setdefault('post_generation_repairs',[]).append(addition)
        if p.name!='generation_checkpoint.json':
            d.update(supports=len(final_rows),permanent_micro_reinforcement_count=sum(r['kind']=='fixed_micro_reinforcement' for r in final_rows),
                permanent_micro_reinforcement_volume_mm3=H.volume(aggregates['holder_micro_reinforcements.npz']),
                support_volume_mm3=H.volume(aggregates['holder_supports.npz']),generation_checks_passed=all(r['passed'] for r in final_rows),passed=False)
        name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
        (STAGE/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    require(all(H.sha(ROOT/p)==digest for p,digest in hashes.items()),'Protected input changed before commit')
    require(all(H.sha(OUT/p)==digest for p,digest in old_mesh_hashes.items()),'An existing support mesh changed')
    if a.apply:
        for name in generated+list(aggregates):(STAGE/name).replace(OUT/name)
        for p in paths:
            name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
            (STAGE/name).replace(p)
        require(all(H.sha(OUT/p)==digest for p,digest in old_mesh_hashes.items()),'Existing support mesh changed during commit')
        require(H.sha(ROOT/'cache/original_v5.npz')==hashes['cache/original_v5.npz'],'Original model changed')
        require(all(H.sha(OUT/name)==digest for name,digest in aggregate_hashes.items()),'Committed aggregate hash mismatch')
    REPORT.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ['pre_integration_input_sha256','existing_declared_mesh_sha256']},indent=2),flush=True)

if __name__=='__main__':main()
