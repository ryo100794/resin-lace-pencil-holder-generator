#!/usr/bin/env python3
"""Package the verified V6 release, excluding unused support prototypes."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'
LIMIT=100*1024*1024
SOURCE_EXCLUDE={
    'build_coupon_v6.py','repair_holder_tip_seams_v6.py',
    'custom_sphere_supports_canonical_diagnostic.py','custom_sphere_supports_component_diagnostic.py',
    'custom_sphere_supports_edge_attribution.py','custom_sphere_supports_fresh_trial.py',
    'custom_sphere_supports_simplify_diagnostic.py','custom_sphere_supports_whole_diagnostic.py',
    'custom_sphere_supports_frozen_birth_audit.py',
}
REPORT_INVENTORY={
    # Relevant immutable V5 mechanism and precision provenance.
    'design.json','export_validation.json','geometry_validation.json',
    'holder_motion_clearance.json','holder_service_access.json','motion_validation.json',
    'moving_envelopes.json','moving_envelopes_details.json','precision_repairs_scope.json',
    'print_preparation.json','sphere_generation_checks.json','strength_validation.json',
    'early_support_validation.json','full_height_validation.json','support_projection_clearance.json',
    # Actual V6 inputs, final gates, and the like-for-like original comparison.
    'support_analysis_v5_baseline.json','support_analysis_classifier_validation.json',
    'support_analysis_fixed_holder_contacts.json','v5_mechanism_unchanged.json',
    'custom_holder_supports.json','custom_sphere_supports_3.json','custom_sphere_supports_4.json',
    'custom_sphere_supports_5.json','custom_sphere_supports_manifest.json',
    'custom_sphere_supports_cross_contacts.json','rotation_envelope_certificate.json',
    'custom_support_assembly.json','custom_supports_validation.json',
    'v6_precision_preparation.json','v6_precision_motion_validation.json',
    'v6_external_removal_validation.json','v6_internal_service_validation.json',
    'v6_export_validation.json','v6_release_validation.json','v6_delivery_summary.json',
    'custom_support_render.json',
    'fixed_precision_source_attribution.json','fixed_precision_repair_candidates.json',
    'fixed_precision_repair_scope_checks.json',
    'fixed_precision_repair_integration.json','custom_sphere_supports_joint_repairs.json',
    'rotation_envelope_sphere_additions.json',
    'v6_raw_component_diagnostic.json',
    'v6_nominal_design_context.json',
    'fixed_precision_additional_rim_candidates.json','fixed_precision_additional_rim_scope_checks.json',
    'fixed_precision_additional_rim_integration.json',
    'fixed_precision_robust_rim_candidates.json','fixed_precision_robust_rim_scope_checks.json',
    'fixed_precision_robust_rim_current_context.json','fixed_precision_robust_rim_integration.json',
    'sphere_removal_partition_v6.json','sphere_removal_partition_local_context_v6.json',
    'fixed_precision_point_boxes_candidates.json','fixed_precision_point_boxes_scope_checks.json',
    'fixed_precision_point_boxes_current_context.json','fixed_precision_point_boxes_integration.json',
    'final_rim_start_attribution.json','rim_cap_bedding_candidates_v6.json',
    'rim_cap_bedding_scope_checks_v6.json','fixed_precision_cap_bedding_integration.json',
    'rim_cap_bedding_history_inventory_v6.json',
    'rim_cap_corner_expansions_candidates_v6.json','rim_cap_corner_expansions_scope_v6.json',
    'fixed_precision_cap_corner_integration.json',
    'fixed_precision_cap_corner_history.json',
    'final_rounded_cap_candidates_v6.json','final_rounded_cap_scope_v6.json',
    'fixed_precision_final_cap_integration.json',
    'holder_full_aggregate_recovery_v6.json',
    'residual_cap_prefix_validation_v6.json','residual_rounded_caps_candidates_v6.json',
    'residual_rounded_caps_scope_v6.json','fixed_precision_residual_caps_integration.json',
    'external_removal_plate_stock_small_mesh_validation.json',
    'plate_stock_literal_noop_diagnostic_v6.json',
    'plate_stock_b2_preflight_v6.json',
    'holder_finite_endpoint_partition_v6.json',
    'external_removal_prisms_b3_fixture_validation.json',
    'external_removal_prisms_b3_small_body_validation.json',
    'external_removal_prism_stream_equivalence_v6.json',
    'external_removal_collision_attribution_cac74327_v6.json',
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def gate(name):
    d=json.loads((ROOT/'reports'/name).read_text())
    if d.get('passed') is not True:raise ValueError('Cannot package: '+name)
    return d

def referenced_meshes(value,folder):
    found=set()
    if isinstance(value,dict):
        for v in value.values():found|=referenced_meshes(v,folder)
    elif isinstance(value,list):
        for v in value:found|=referenced_meshes(v,folder)
    elif isinstance(value,str) and value.endswith('.npz'):
        p=folder/value
        if not p.exists():p=ROOT/value
        if p.exists() and p.is_file() and 'support_environment' not in p.parts:found.add(p)
    return found

def referenced_json_evidence(value,folder,seen=None):
    """Keep explicit adopted-repair evidence without globbing old failures."""
    seen=set() if seen is None else seen
    values=value.values() if isinstance(value,dict) else value if isinstance(value,list) else [value]
    for child in values:
        if isinstance(child,(dict,list)):
            referenced_json_evidence(child,folder,seen)
        elif isinstance(child,str) and child.endswith('.json'):
            p=folder/child
            if not p.exists():p=ROOT/child
            p=p.resolve()
            if p.exists() and p.is_relative_to(ROOT) and p not in seen:
                seen.add(p)
                referenced_json_evidence(json.loads(p.read_text()),p.parent,seen)
    return seen

def check_zip(path):
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        count=len(z.infolist())
    if path.stat().st_size>LIMIT:raise ValueError('Drive upload size exceeds 100 MiB: '+str(path))
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),members=count,crc_passed=True)

def is_support_asset(path):
    """Ordinary companion ZIP; both archives keep the same project/ root."""
    relative=path.relative_to(ROOT)
    if not relative.parts or relative.parts[0]!='cache':return False
    if len(relative.parts)>1 and relative.parts[1] in {
            'custom_holder_supports','custom_sphere_supports',
            'sphere_removal_partition_v6','external_removal_v6',
            'holder_finite_endpoint_partition_v6',
            'external_removal_plate_stock_v6',
            'external_removal_plate_stock_v6_b2',
            'fixed_precision_repairs','support_environment',
            'rim_cap_bedding_v6','residual_rounded_caps_v6','residual_cap_births_v6'}:
        return True
    return relative.name.startswith('added_') and relative.suffix=='.npz'

def archive_name(path):
    return 'project/'+(path.name if path.parent==OUT else path.relative_to(ROOT).as_posix())

def main():
    # Recheck every current manifest binding, including actual delivered bytes.
    subprocess.run([sys.executable,str(ROOT/'source/audit_v6_release.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'source/write_v6_delivery_notes.py')],cwd=ROOT,check=True)
    required=['custom_support_assembly.json','custom_supports_validation.json',
        'rotation_envelope_certificate.json','v6_external_removal_validation.json',
        'v6_internal_service_validation.json','v6_export_validation.json',
        'v5_mechanism_unchanged.json','v6_delivery_summary.json',
        'v6_precision_motion_validation.json','v6_release_validation.json','custom_support_render.json']
    for name in required:gate(name)
    files={p for p in (ROOT/'source').glob('*.py') if p.name not in SOURCE_EXCLUDE}
    files|={ROOT/'source/requirements.txt',ROOT/'cache/original_v5.npz'}
    history_files=set()
    files|={p for p in (ROOT/'cache/spheres').iterdir() if p.is_file()}
    files|={p for p in (ROOT/'cache/rotation_envelopes').glob('*') if p.is_file()}
    files|={p for p in (ROOT/'cache/fixed_precision_repairs').rglob('*') if p.is_file()}
    folder=ROOT/'cache/custom_holder_supports';hp=folder/'holder_supports.json'
    holder=json.loads(hp.read_text());files.add(hp);files|=referenced_meshes(holder,folder)
    files|=referenced_json_evidence(holder.get('post_generation_repairs',[]),folder)
    for n in (3,4,5):
        folder=ROOT/'cache/custom_sphere_supports';sp=folder/f'sphere_{n}layers_supports.json'
        sphere=json.loads(sp.read_text());files.add(sp);files|=referenced_meshes(sphere,folder)
        files|=referenced_json_evidence(sphere.get('post_generation_removable_body_repairs',[]),folder)
        for row in sphere['supports']:
            files|=referenced_json_evidence(row.get('removable_body_repairs',[]),folder)
            if row.get('removable_body_repairs'):
                for suffix in ['collar','full']:
                    adopted=folder/'joint_repairs_diagnostic'/f'sphere_{n}layers_support_{row["support_index"]:03d}_root_join_{suffix}.npz'
                    if not adopted.exists():raise ValueError('Missing adopted collar history: '+str(adopted))
                    history_files.add(adopted)
        envelope_metadata=folder/f'sphere_{n}layers_motion_envelope.json'
        files.add(envelope_metadata)
        source_hashes={sphere.get('generation_source_sha256')}
        source_hashes.update(r.get('generation_source_sha256') for r in sphere['supports'])
        for digest in source_hashes:
            if digest:
                snapshot=folder/f'generator_source_{digest}.py'
                if snapshot.exists():files.add(snapshot)
        # The supported-outer union is a regenerable local diagnostic cache;
        # the complete current model uses original_v5 plus the support parts.
        for suffix in ['supports','retained_stumps','micro_reinforcements']:
            q=folder/f'sphere_{n}layers_{suffix}.npz'
            if q.exists():files.add(q)
        for r in sphere['supports']:
            for k in range(2):
                q=folder/f'sphere_{n}layers_neck_{r["support_index"]:03d}_{k}.npz'
                if q.exists():files.add(q)
    for name in ['holder_supports.npz','holder_retained_stumps.npz','holder_micro_reinforcements.npz']:
        files.add(ROOT/'cache/custom_holder_supports'/name)
    external=gate('v6_external_removal_validation.json')
    files|=referenced_meshes(external,ROOT)
    # Preserve every actual routing input, including the compact instruction-
    # only candidate JSON and finite-cut manifest, not just mesh filenames.
    for row in external['input_files']:
        p=ROOT/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:
            raise ValueError('Changed removal input: '+row['path'])
        files.add(p)
    # Preserve only the explicit successful B2 preflight records and their
    # referenced meshes, not unrelated stock/prototype cache generations.
    preflight_path=ROOT/'reports/plate_stock_b2_preflight_v6.json'
    preflight_evidence=referenced_json_evidence(json.loads(preflight_path.read_text()),ROOT)
    files.update(preflight_evidence)
    for proof_path in preflight_evidence:
        files|=referenced_meshes(json.loads(proof_path.read_text()),proof_path.parent)
    # Keep released instructions as candidates for a faster reproducible
    # rebuild. No mesh, collision result or success flag is reused.
    route_instructions=[dict(id=row['id'],mode=row['mode'],
        additional_body_cuts=row['additional_body_cuts'],
        fragments=[dict(translation_legs=[dict(direction=leg['direction'],distance_mm=leg['distance_mm'])
                for leg in fragment['translation_legs']],
            body_section_index=fragment.get('body_section_index',index))
            for index,fragment in enumerate(row['fragments'])]) for row in external['plans']]
    if len({row['id'] for row in route_instructions})!=external['initial_support_count']:
        raise ValueError('Released removal instruction IDs are incomplete or duplicated')
    candidate_bytes=(json.dumps(dict(
        purpose='Instructions only. Every cut, continuous movement and exit must be fully revalidated during rebuild.',
        plans=route_instructions),indent=2,allow_nan=False)+'\n').encode()
    candidate_digest=hashlib.sha256(candidate_bytes).hexdigest()
    released_candidates=ROOT/'cache/external_removal_v6'/('released_route_candidates_'+candidate_digest[:16]+'.json')
    if released_candidates.exists():
        if sha(released_candidates)!=candidate_digest:
            raise ValueError('Released instruction snapshot changed')
    else:
        released_candidates.write_bytes(candidate_bytes)
    # The stable index is orchestration metadata, never a planner input. New
    # packaging must not overwrite any candidate file hashed by a prior run.
    candidate_index=ROOT/'reports/v6_rebuild_route_candidates.json'
    candidate_index.write_text(json.dumps(dict(path=str(released_candidates.relative_to(ROOT)),
        sha256=candidate_digest,
        source_removal_validation_sha256=sha(ROOT/'reports/v6_external_removal_validation.json')),
        indent=2)+'\n')
    files.update([released_candidates,candidate_index])
    sphere_repairs=gate('custom_sphere_supports_joint_repairs.json')
    before_manifest=ROOT/sphere_repairs['old_input_backup_manifest']
    files.add(before_manifest)
    history_files.add(before_manifest)
    history_files|={p for p in (before_manifest.parent/'before').rglob('*') if p.is_file()}
    history_files.add(ROOT/'reports/custom_sphere_supports_joint_repairs.json')
    bedding_history_path=ROOT/'reports/rim_cap_bedding_history_inventory_v6.json'
    bedding_history=json.loads(bedding_history_path.read_text())
    history_files.add(bedding_history_path)
    for row in bedding_history['files']:
        p=ROOT/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:
            raise ValueError('Changed fixed-cap history: '+row['path'])
        history_files.add(p)
    for name in REPORT_INVENTORY:
        p=ROOT/'reports'/name
        if not p.exists():raise ValueError('Missing declared release evidence: '+name)
        files.add(p)
    for n in (3,4,5):
        p=ROOT/f'cache/support_environment/sphere_{n}layers_birth_targets.json'
        if p.exists():files.add(p)
    for name in ['README_V6_JA.txt','REMOVE_ORDER_JA.txt','support_removal_index.json']:
        files.add(OUT/name)
    render=gate('custom_support_render.json')
    for key,name in [('original_sha256','original_v5.npz'),
            ('support_mesh_sha256','added_temporary_v6.npz'),('fixed_micro_sha256','added_fixed_micro_v6.npz')]:
        if render[key]!=sha(ROOT/'cache'/name):raise ValueError('Preview source differs from release: '+name)
    preview_files=[OUT/row['file'] for row in render['outputs']]
    for path,row in zip(preview_files,render['outputs']):
        if sha(path)!=row['sha256']:raise ValueError('Rendered preview changed: '+str(path))
    files.update(preview_files)
    manifest=[]
    history_target=OUT/'ivory_v6_repair_history.zip'
    with zipfile.ZipFile(history_target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        history_manifest=[]
        for p in sorted(history_files):
            name='project/'+p.relative_to(ROOT).as_posix();z.write(p,name)
            history_manifest.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
        z.writestr('HISTORY_MANIFEST.json',json.dumps(dict(files=history_manifest,
            print_input=False,scope='Historical accepted joint-repair and fixed-cap bedding evidence. The complete current release is rebuilt from ivory_v6_project.zip plus ivory_v6_support_assets.zip without these backups.',
            extraction='Extract this optional history archive into a separate folder. Do not merge it over a current or rebuilt project; some historical reports share paths.',
            replay_note='Literal historical repair replay additionally requires reconstruction of the historical whole-model diagnostic context, which is not a print deliverable.'),indent=2)+'\n')
    history_package=check_zip(history_target)
    asset_files={p for p in files if is_support_asset(p)}
    main_files=files-asset_files
    asset_target=OUT/'ivory_v6_support_assets.zip'
    asset_manifest=[]
    with zipfile.ZipFile(asset_target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(asset_files):
            if not p.exists():raise ValueError('Missing support asset: '+str(p))
            name=archive_name(p);z.write(p,name)
            row=dict(path=name,bytes=p.stat().st_size,sha256=sha(p),archive=asset_target.name)
            asset_manifest.append(row);manifest.append(row)
        asset_manifest_bytes=(json.dumps(dict(version=6,files=asset_manifest,
            mandatory_companion='ivory_v6_project.zip',
            extraction='Extract both ordinary ZIP archives into the same parent directory, combining their project/ folders.',
            scope='Adopted support parts, finite-cut partitions, routing fragments and their frozen source data.'),indent=2)+'\n').encode()
        z.writestr('project/ASSET_MANIFEST.json',asset_manifest_bytes)
    asset_package=check_zip(asset_target)
    asset_manifest_sha256=hashlib.sha256(asset_manifest_bytes).hexdigest()
    source_target=OUT/'ivory_v6_project.zip'
    with zipfile.ZipFile(source_target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(main_files):
            if not p.exists():raise ValueError('Missing project input: '+str(p))
            name=archive_name(p)
            z.write(p,name)
            manifest.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p),archive=source_target.name))
        z.writestr('project/FILE_MANIFEST.json',json.dumps(dict(version=6,files=sorted(manifest,key=lambda row:row['path']),
            immutable_v5_source='cache/original_v5.npz',
            reproduction='python source/rebuild_v6.py',
            required_support_assets_archive=asset_package,
            required_asset_manifest_sha256=asset_manifest_sha256,
            extraction='Extract ivory_v6_project.zip and ivory_v6_support_assets.zip into the same parent directory, combining their project/ folders. Both archives are mandatory for rebuilding.',
            optional_repair_history_archive=history_package,
            notes='Only declared final support parts are active inputs. Explicit adopted-repair evidence and immutable V5 provenance are retained. Unrelated failed geometry trials, stale checkpoints, raw context unions and working previews are excluded.'),indent=2)+'\n')
    stl=OUT/'ivory_penholder_v6_custom_supported.stl'
    stl_zip=OUT/'ivory_v6_stl.zip'
    with zipfile.ZipFile(stl_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.write(stl,stl.name);z.write(OUT/'README_V6_JA.txt','README_V6_JA.txt')
    packages=[check_zip(source_target),asset_package,check_zip(stl_zip),history_package]
    primary=OUT/'ivory_penholder_v6_custom_supported.3mf'
    if primary.stat().st_size>LIMIT:raise ValueError('3MF exceeds Drive file size limit')
    published=[primary,stl_zip,source_target,asset_target,history_target,OUT/'README_V6_JA.txt',OUT/'REMOVE_ORDER_JA.txt',
        OUT/'support_removal_index.json']+preview_files
    for path in published:
        if path.stat().st_size>LIMIT:
            raise ValueError('Publish file exceeds Drive upload limit; package it separately: '+path.name)
    sums=''.join(sha(p)+'  '+p.name+'\n' for p in published)
    (OUT/'SHA256SUMS.txt').write_text(sums)
    published.append(OUT/'SHA256SUMS.txt')
    result=dict(passed=True,version=6,project_files=len(manifest),
        main_project_files=len(main_files),support_asset_files=len(asset_files),
        required_rebuild_archives=[source_target.name,asset_target.name],packages=packages,
        publish_files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in published],
        scope='ZIP CRC, declared member inventory and content hashes. Actual Drive readback is a separate delivery check.')
    (ROOT/'reports/v6_package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,project_files=len(manifest),packages=packages)),flush=True)

if __name__=='__main__':main()
