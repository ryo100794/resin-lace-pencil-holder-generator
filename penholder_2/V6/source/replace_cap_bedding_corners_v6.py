#!/usr/bin/env python3
"""Adopt the three independently checked fixed cap-corner positive supersets.

Only H1288/H1294/H1307 full and retained meshes change. Their entire prior solids,
all original V5 geometry, and all temporary/cut stock are preserved.
"""
from pathlib import Path
import argparse,copy,json,shutil,time
import custom_holder_supports as H

ROOT=H.ROOT;OUT=H.OUT
CANDIDATES=None
SCOPE=None
REPORT=ROOT/'reports/fixed_precision_cap_corner_integration.json'
STAGE=OUT/'cap_corner_integration'
EXPECTED_FULL='7b4f58873ab6ce3fd02eaa8819949d0069a492eac193b4aeb33189e3bdf68c78'
EXPECTED_CANDIDATE=None
INDICES={1288,1294,1307}

def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    global CANDIDATES,SCOPE,EXPECTED_CANDIDATE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--scope',type=Path,required=True)
    ap.add_argument('--candidate-sha256',required=True)
    ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    CANDIDATES=a.candidates.resolve();SCOPE=a.scope.resolve();EXPECTED_CANDIDATE=a.candidate_sha256
    require(CANDIDATES.is_relative_to(ROOT) and SCOPE.is_relative_to(ROOT),'Evidence outside project')
    start=time.monotonic();paths=[OUT/'holder_supports.json',ROOT/'reports/custom_holder_supports.json',OUT/'generation_checkpoint.json']
    documents=[json.loads(p.read_text()) for p in paths];rows=documents[0]['parts']
    if len(rows)==1330 and all(rows[i].get('precision_fixed_fill',{}).get('candidate_manifest_sha256')==EXPECTED_CANDIDATE for i in INDICES):
        print('The three exact fixed cap-corner replacements are already integrated.',flush=True);return
    require(len(rows)==1330 and all(d['parts']==rows for d in documents),'Unexpected holder manifests')
    require(H.sha(OUT/'holder_supports.npz')==EXPECTED_FULL,'Unexpected holder aggregate')
    require(H.sha(CANDIDATES)==EXPECTED_CANDIDATE,'Unexpected candidate manifest')
    candidates=json.loads(CANDIDATES.read_text());scope=json.loads(SCOPE.read_text())
    require(candidates['local_geometry_passed'] and len(candidates['repairs'])==3 and scope['passed'],'Independent repair gates failed')
    require({r['replacement_holder_index'] for r in candidates['repairs']}==INDICES,'Unexpected replacement set')
    stale=[p for p,d in scope['input_sha256'].items() if H.sha(ROOT/p)!=d]
    require(not stale,'Scope inputs changed: '+str(stale))
    protected=[ROOT/'cache/original_v5.npz',CANDIDATES,SCOPE,*paths,
        OUT/'holder_supports.npz',OUT/'holder_retained_stumps.npz',OUT/'holder_micro_reinforcements.npz']
    hashes={str(p.relative_to(ROOT)):H.sha(p) for p in protected}
    mutable={rows[i][key] for i in INDICES for key in ['full_mesh_file','retained_mesh_file']}
    unchanged={name:H.sha(OUT/name) for r in rows for name in [r['full_mesh_file'],r['retained_mesh_file'],r['removable_mesh_file']]
        if name not in mutable}
    STAGE.mkdir(exist_ok=True);before=STAGE/'before';before.mkdir(exist_ok=True)
    for p in paths:shutil.copy2(p,before/('reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name))
    final_rows=copy.deepcopy(rows);replacements={};checks={r['id']:r for r in scope['repairs']}
    for candidate in candidates['repairs']:
        index=candidate['replacement_holder_index'];check=checks[candidate['id']];old=rows[index]
        require(old['index']==index and old['kind']=='fixed_micro_reinforcement','Wrong replacement owner')
        require(check['passed'] and check['persistent_fixed_contact']['passed'],'Unrooted replacement')
        source=ROOT/candidate['mesh_file'];require(H.sha(source)==candidate['mesh_sha256']==check['mesh_sha256'],'Changed candidate')
        prior=H.load(OUT/old['full_mesh_file'])[0];new=H.load(source)[0]
        require(H.sha(OUT/old['full_mesh_file'])==candidate['old_piece_sha256'],'Old adopted piece changed')
        require((prior-new).is_empty(),'Replacement removes prior material')
        replacements[index]=new
        for key in ['full_mesh_file','retained_mesh_file']:
            shutil.copy2(OUT/old[key],before/old[key]);shutil.copy2(source,STAGE/old[key])
        row=final_rows[index]
        row.update(root_mm=candidate['root_mm'],target_mm=candidate['target_mm'],bounds_mm=candidate['bounds_mm'],volume_mm3=candidate['volume_mm3'],
            actual_added_volume_over_occupied_context_mm3=candidate['actual_added_volume_over_occupied_context_mm3'],
            root_contact_volume_mm3=check['persistent_fixed_contact']['contact_volume_mm3'],
            moving_envelope_clearance_mm=check['categories']['continuous_motion_envelope']['clearance_lower_bound_mm'],
            generation_source_sha256=candidates['source_sha256'])
        row['superseded_precision_fixed_fill']=copy.deepcopy(old['precision_fixed_fill'])
        row['precision_fixed_fill']=dict(id=candidate['id'],purpose=candidate['repair_purpose'],
            candidate_manifest=str(CANDIDATES.relative_to(ROOT)),candidate_manifest_sha256=H.sha(CANDIDATES),
            source_mesh_file=candidate['mesh_file'],source_mesh_sha256=candidate['mesh_sha256'],
            pre_integration_scope_report=str(SCOPE.relative_to(ROOT)),pre_integration_scope_report_sha256=H.sha(SCOPE),
            persistent_fixed_contact=check['persistent_fixed_contact'],local_after_strict_probe=candidate['local_after_strict_probe'],
            complete_old_piece_contained=True,old_piece_file=candidate['old_piece_file'],old_piece_sha256=candidate['old_piece_sha256'],
            source_evidence=copy.deepcopy(candidate),global_build_and_external_removal_required=True)
    fulls=[replacements[r['index']] if r['index'] in replacements else H.load(OUT/r['full_mesh_file'])[0] for r in rows]
    retained=[replacements[r['index']] if r['index'] in replacements else H.load(OUT/r['retained_mesh_file'])[0] for r in rows]
    aggregates={'holder_supports.npz':H.union(fulls),'holder_retained_stumps.npz':H.union(retained),
        'holder_micro_reinforcements.npz':H.union([p for p,r in zip(fulls,final_rows) if r['kind']=='fixed_micro_reinforcement'])}
    for name,mesh in aggregates.items():H.save(STAGE/name,mesh)
    aggregate_hashes={name:H.sha(STAGE/name) for name in aggregates}
    proof=dict(passed=True,applied=a.apply,part_count_before=1330,part_count_after=1330,replaced_indices=sorted(INDICES),
        all_other_declared_meshes_unchanged=True,immutable_original_unchanged=True,all_temporary_and_cut_stock_unchanged=True,
        complete_prior_material_preserved=True,pre_integration_input_sha256=hashes,
        unchanged_declared_mesh_sha256=unchanged,aggregate_sha256=aggregate_hashes,
        superseded_pieces=[dict(path=r['old_piece_file'],sha256=r['old_piece_sha256']) for r in candidates['repairs']],
        actual_added_volume_over_occupied_context_mm3=sum(r['actual_added_volume_over_occupied_context_mm3'] for r in candidates['repairs']),
        whole_replacement_pieces_volume_mm3=sum(r['volume_mm3'] for r in candidates['repairs']),
        scope_refresh_required_after_integration=True,final_whole_model_gates_required=True,
        integration_source_sha256=H.sha(__file__),elapsed_seconds=time.monotonic()-start)
    addition=dict(kind='positive_superset_fixed_cap_corner_replacements',replaced_part_indices=sorted(INDICES),
        report=str(REPORT.relative_to(ROOT)),passed=True,candidate_manifest=str(CANDIDATES.relative_to(ROOT)),
        scope_report=str(SCOPE.relative_to(ROOT)),
        source_sha256=H.sha(__file__))
    for d,p in zip(documents,paths):
        d['parts']=copy.deepcopy(final_rows);d.setdefault('post_generation_repairs',[]).append(addition)
        if p.name!='generation_checkpoint.json':
            d.update(permanent_micro_reinforcement_volume_mm3=H.volume(aggregates['holder_micro_reinforcements.npz']),
                support_volume_mm3=H.volume(aggregates['holder_supports.npz']),generation_checks_passed=all(r['passed'] for r in final_rows),passed=False)
        name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
        (STAGE/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    require(all(H.sha(ROOT/p)==digest for p,digest in hashes.items()),'Protected input changed before commit')
    require(all(H.sha(OUT/p)==digest for p,digest in unchanged.items()),'Another support mesh changed')
    if a.apply:
        for name in [*sorted(mutable),*aggregates]:(STAGE/name).replace(OUT/name)
        for p in paths:
            name='reports_custom_holder_supports.json' if p.parent.name=='reports' else p.name
            (STAGE/name).replace(p)
        require(all(H.sha(OUT/p)==digest for p,digest in unchanged.items()),'Another support mesh changed during commit')
        require(H.sha(ROOT/'cache/original_v5.npz')==hashes['cache/original_v5.npz'],'Original model changed')
    REPORT.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ['pre_integration_input_sha256','unchanged_declared_mesh_sha256']},indent=2),flush=True)

if __name__=='__main__':main()
