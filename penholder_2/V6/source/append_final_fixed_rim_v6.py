#!/usr/bin/env python3
"""Append one independently guarded permanent rim fill as H1284.

Every earlier printed holder piece and the immutable V5 mesh is preserved.
This is an explicit addition, not deletion of the small contained rim lobe.
"""
from pathlib import Path
import argparse,copy,json,shutil,time
import manifold3d as mf
import custom_holder_supports as H

ROOT=H.ROOT;OUT=H.OUT
CANDIDATES=ROOT/'reports/fixed_precision_additional_rim_candidates.json'
SCOPE=ROOT/'reports/fixed_precision_additional_rim_scope_checks.json'
REPORT=ROOT/'reports/fixed_precision_additional_rim_integration.json'
STAGE=OUT/'final_rim_integration'
EXPECTED_FULL='51db444e0442be36ee7e3d50895a28970970970b26a30548011296b2177c256d'
EXPECTED_CANDIDATE='cf91fe944134c750c8649fdeedcbe8b46ac61b770a19fcdf239ea067d47f1ccb'

def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    start=time.monotonic()
    paths=[OUT/'holder_supports.json',ROOT/'reports/custom_holder_supports.json',OUT/'generation_checkpoint.json']
    documents=[json.loads(p.read_text()) for p in paths];rows=documents[0]['parts']
    if len(rows)==1285 and rows[-1].get('precision_fixed_fill',{}).get('id')=='rim_final':
        print('H1284 is already integrated.',flush=True);return
    require(len(rows)==1284 and all(d['parts']==rows for d in documents),'Unexpected holder manifests')
    require(H.sha(OUT/'holder_supports.npz')==EXPECTED_FULL,'Unexpected holder aggregate')
    require(H.sha(CANDIDATES)==EXPECTED_CANDIDATE,'Unexpected candidate manifest')
    candidates=json.loads(CANDIDATES.read_text());scope=json.loads(SCOPE.read_text())
    require(candidates['local_geometry_passed'] and len(candidates['repairs'])==1,'Candidate local gate failed')
    require(scope.get('passed') is True and len(scope['repairs'])==1,'Candidate exclusion gate failed')
    stale=[p for p,d in scope['input_sha256'].items() if H.sha(ROOT/p)!=d]
    require(not stale,'Scope inputs changed: '+str(stale))
    candidate=candidates['repairs'][0];check=scope['repairs'][0]
    require(candidate['id']=='rim_final' and check['id']=='rim_final' and check['passed'],'Unexpected fixed patch')
    require(check['persistent_fixed_contact']['passed'],'Persistent attachment failed')
    source=ROOT/candidate['mesh_file'];require(H.sha(source)==candidate['mesh_sha256']==check['mesh_sha256'],'Changed candidate')
    full=H.load(source)[0]
    protected=[ROOT/'cache/original_v5.npz',CANDIDATES,SCOPE,*paths,
        OUT/'holder_supports.npz',OUT/'holder_retained_stumps.npz',OUT/'holder_micro_reinforcements.npz']
    hashes={str(p.relative_to(ROOT)):H.sha(p) for p in protected}
    old_mesh_hashes={name:H.sha(OUT/name) for r in rows for name in
        [r['full_mesh_file'],r['retained_mesh_file'],r['removable_mesh_file']]}
    STAGE.mkdir(exist_ok=True);before=STAGE/'before';before.mkdir(exist_ok=True)
    for p in paths:shutil.copy2(p,before/('reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name))
    names={k:f'holder_support_1284_{suffix}.npz' for k,suffix in
        [('full_mesh_file','full'),('retained_mesh_file','retained'),('removable_mesh_file','removable')]}
    for k in ['full_mesh_file','retained_mesh_file']:shutil.copy2(source,STAGE/names[k])
    H.save(STAGE/names['removable_mesh_file'],mf.Manifold())
    row=dict(index=1284,kind='fixed_micro_reinforcement',temporary=False,
        root_mm=candidate['root_mm'],target_mm=candidate['target_mm'],bounds_mm=candidate['bounds_mm'],
        volume_mm3=candidate['volume_mm3'],actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'],
        root_contact_volume_mm3=check['persistent_fixed_contact']['contact_volume_mm3'],
        moving_envelope_clearance_mm=check['categories']['continuous_motion_envelope']['clearance_lower_bound_mm'],
        neck_count=0,removal='Permanent fixed-holder precision fill; do not cut.',
        generation_source_sha256=candidates['source_sha256'],passed=True,
        precision_fixed_fill=dict(id='rim_final',purpose=candidate['repair_purpose'],
            candidate_manifest=str(CANDIDATES.relative_to(ROOT)),candidate_manifest_sha256=H.sha(CANDIDATES),
            source_mesh_file=candidate['mesh_file'],source_mesh_sha256=candidate['mesh_sha256'],
            pre_integration_scope_report=str(SCOPE.relative_to(ROOT)),pre_integration_scope_report_sha256=H.sha(SCOPE),
            persistent_fixed_contact=check['persistent_fixed_contact'],
            local_after_strict_probe=candidate['local_after_strict_probe'],
            preserved_positive_lobe_volume_mm3=candidate['preserved_positive_lobe_volume_mm3'],
            entire_source_lobe_contained_in_patch=candidate['entire_source_lobe_contained_in_patch'],
            global_build_and_external_removal_required=True),**names)
    final_rows=copy.deepcopy(rows)+[row]
    fulls=[H.load(OUT/r['full_mesh_file'])[0] for r in rows]+[full]
    retained=[H.load(OUT/r['retained_mesh_file'])[0] for r in rows]+[full]
    aggregates={'holder_supports.npz':H.union(fulls),'holder_retained_stumps.npz':H.union(retained),
        'holder_micro_reinforcements.npz':H.union([p for p,r in zip(fulls,final_rows) if r['kind']=='fixed_micro_reinforcement'])}
    for name,mesh in aggregates.items():H.save(STAGE/name,mesh)
    aggregate_hashes={name:H.sha(STAGE/name) for name in aggregates}
    proof=dict(passed=True,applied=args.apply,part_count_before=1284,part_count_after=1285,added_indices=[1284],
        all_previous_declared_meshes_unchanged=True,immutable_original_unchanged=True,
        pre_integration_input_sha256=hashes,existing_declared_mesh_sha256=old_mesh_hashes,
        aggregate_sha256=aggregate_hashes,actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'],
        whole_piece_volume_mm3=candidate['volume_mm3'],source_lobe_preserved=True,
        scope_refresh_required_after_integration=True,final_whole_model_gates_required=True,
        integration_source_sha256=H.sha(__file__),elapsed_seconds=time.monotonic()-start)
    addition=dict(kind='explicit_final_fixed_rim_fill',added_part_indices=[1284],
        report=str(REPORT.relative_to(ROOT)),passed=True,
        candidate_manifest=str(CANDIDATES.relative_to(ROOT)),scope_report=str(SCOPE.relative_to(ROOT)),
        source_sha256=H.sha(__file__),actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'])
    for d,p in zip(documents,paths):
        d['parts']=copy.deepcopy(final_rows);d.setdefault('post_generation_repairs',[]).append(addition)
        if p.name!='generation_checkpoint.json':
            d.update(supports=len(final_rows),permanent_micro_reinforcement_count=sum(r['kind']=='fixed_micro_reinforcement' for r in final_rows),
                permanent_micro_reinforcement_volume_mm3=H.volume(aggregates['holder_micro_reinforcements.npz']),
                support_volume_mm3=H.volume(aggregates['holder_supports.npz']),generation_checks_passed=all(r['passed'] for r in final_rows),passed=False)
        name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
        (STAGE/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    require(all(H.sha(ROOT/p)==d for p,d in hashes.items()),'Protected input changed before commit')
    require(all(H.sha(OUT/p)==d for p,d in old_mesh_hashes.items()),'An earlier mesh changed')
    if args.apply:
        for name in [*names.values(),*aggregates]:(STAGE/name).replace(OUT/name)
        for p in paths:
            name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
            (STAGE/name).replace(p)
        require(all(H.sha(OUT/p)==d for p,d in old_mesh_hashes.items()),'Earlier mesh changed during commit')
        require(H.sha(ROOT/'cache/original_v5.npz')==hashes['cache/original_v5.npz'],'Original source changed')
    REPORT.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ['pre_integration_input_sha256','existing_declared_mesh_sha256']},indent=2),flush=True)

if __name__=='__main__':main()
