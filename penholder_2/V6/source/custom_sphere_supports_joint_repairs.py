#!/usr/bin/env python3
"""Scoped positive-volume root-carrier joins in removable sphere supports.

All original V5 meshes and every old neck/stump mesh remain untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import manifold3d as mf
import numpy as np
import build_test_coupon as B
import custom_sphere_supports as S
from build_holder import SPHERES, sphere_transform

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/custom_sphere_supports'
DIAG=OUT/'joint_repairs_diagnostic'
IDS={3:[15,29,31],4:[18,35,48,49],5:[9,16,17,48]}
RETAINED_ONLY_IDS={4:[22,23,31],5:[18,19,22,29]}
EPS=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict(m):
    try:
        v,f,r=B.canonical_float32_mesh(m.simplify(1e-5));return dict(passed=True,report=r)
    except Exception as error:
        tb=error.__traceback__;data={}
        while tb:
            if tb.tb_frame.f_code.co_name=='canonical_float32_mesh':data=dict(tb.tb_frame.f_locals);break
            tb=tb.tb_next
        result=dict(passed=False,error=str(error),bad_edges=data.get('bad_edges'),zero_area=data.get('zero_area'),near_zero=data.get('near_zero'))
        if 'components' in data:result['component_volumes_mm3']=[float(x.volume()) for x in data['components']]
        return result


def internal_keepouts(meta,cache):
    moving=S.join([S.load(cache/p) for p in meta['free_layer_meshes'][1:]]+[S.load(cache/w['mesh_file']) for w in meta['release_wires']])
    tools=S.join([S.load(cache/e['tool_corridor_mesh_file']) for w in meta['release_wires'] for e in w['endpoints']])
    extraction=[]
    for wire in meta['release_wires']:
        m=S.load(cache/wire['removable_mesh_file']);bb=m.bounding_box();cs=mf.CrossSection(m.project().to_polygons(),mf.FillRule.Positive)
        extraction.append(mf.Manifold.extrude(cs,meta['outer_nominal_radius_mm']+4-bb[2]).translate((0,0,bb[2])))
    return moving,tools,S.join(extraction)


def prepare(n,index,context=None,holder_full=None):
    report=json.loads((OUT/f'sphere_{n}layers_supports.json').read_text());rec=next(x for x in report['supports'] if x['support_index']==index)
    if rec['kind']!='removable_fixed_support':raise RuntimeError('Only a temporary body may receive this collar')
    full=S.load(OUT/rec['mesh_file']);st=S.load(OUT/rec['stump_mesh_file']);maximum=S.load(OUT/rec['conservative_retained_mesh_file'])
    neck_paths=[OUT/f'sphere_{n}layers_neck_{index:03d}_{j}.npz' for j in range(2)]
    fixed_paths=[OUT/rec['stump_mesh_file'],OUT/rec['conservative_retained_mesh_file']]+neck_paths
    hashes={str(p.relative_to(ROOT)):sha(p) for p in fixed_paths}
    end=rec['end_profiles'][0];q=np.asarray(end['point']);d=np.asarray(end['direction']);d=d/np.linalg.norm(d);carrier,radius=end['stations'][-1];center=q+d*carrier
    service_clip=None
    if (n,index)==(5,16):
        # The previous temporary branch crosses wire 10's cutting/extraction
        # corridors. Its leftmost carrier seam can be covered outside both by
        # retaining the -X side of the actual extraction projection's support
        # plane. A single halfspace preserves convexity of the remaining collar.
        wire_path=ROOT/'cache/spheres/sphere_R13_L5_wire_10_removable.npz'
        wire=S.load(wire_path);projection=mf.CrossSection(wire.project().to_polygons(),mf.FillRule.Positive)
        minimum_x=float(min(np.min(poly[:,0]) for poly in projection.to_polygons()))
        normal=np.array([-1.,0.,0.]);offset=-minimum_x+1e-5
        service_clip=dict(wire_path=str(wire_path.relative_to(ROOT)),wire_sha256=sha(wire_path),
            basis='-X supporting halfspace of actual positive-fill XY wire-removal projection',
            kept_halfspace_normal=normal.tolist(),kept_halfspace_offset_mm=offset,geometric_margin_mm=1e-5)
    def clip_service(mesh):
        return mesh.trim_by_plane(service_clip['kept_halfspace_normal'],service_clip['kept_halfspace_offset_mm']) if service_clip else mesh
    seating_attempts=[]
    for bottom,inset in [(-.025,.003),(-.035,.006),(-.038,.008)]:
        stations=[(bottom,radius-inset),(-.010,radius+.001),(.010,radius+.001),(.020,radius-.003)]
        patch=clip_service(S.profile(center,d,stations))
        convex_error=S.vol(mf.Manifold.hull_points(S.verts(patch))-patch)
        if convex_error>EPS:raise RuntimeError('Collar is not convex')
        # The first section must be seated in actual saved stock, even when an
        # old Boolean wedge differs from the original ideal carrier profile.
        slope=(inset+.001)/(-.010-bottom)
        first=clip_service(S.profile(center,d,[stations[0],(bottom+.0001,radius-inset+.0001*slope)]))
        first_outside=S.vol(first-full)
        low=S.verts(patch);minimum=float(low[:,1].min());min_vertices=low[low[:,1]<=minimum+1e-9]
        initial_vertex_checks=[]
        for vertex in min_vertices:
            outside=S.vol(S.ball(vertex,.0001)-full)
            initial_vertex_checks.append(dict(point_mm=vertex.tolist(),probe_radius_mm=.0001,outside_original_support_mm3=outside))
        seating_attempts.append(dict(bottom_axial_mm=bottom,inset_radius_mm=inset,first_section_outside_old_mm3=first_outside,
            maximum_initial_vertex_probe_outside_mm3=max(x['outside_original_support_mm3'] for x in initial_vertex_checks)))
        if first_outside<=1e-14 and all(x['outside_original_support_mm3']<=1e-16 for x in initial_vertex_checks):break
    else:raise RuntimeError('No safely seated inset collar: '+json.dumps(seating_attempts))
    new=(full+patch).as_original();delta=patch-full
    prefix_checks=[]
    for height in S.schedule(patch,.01):
        for part in S.prefix(patch,height).decompose():
            if S.vol(part)<=EPS:continue
            contact=S.overlap(part,full)
            prefix_checks.append(dict(local_y_mm=height,part_volume_mm3=S.vol(part),earlier_carrier_contact_mm3=contact))
            if contact<=EPS:raise RuntimeError(f'Unseated collar prefix at {height}')
    tests={
        'nominal_stump_overlap':S.overlap(patch,st),
        'maximum_stump_overlap':S.overlap(patch,maximum),
        'neck_overlap':max(S.overlap(patch,S.load(p)) for p in neck_paths),
        'old_material_removed':S.vol(full-new),
        'patch_outside_new_nominal_removable':S.vol(patch-(new-st)),
    }
    cache=ROOT/'cache/spheres';meta=json.loads((cache/f'sphere_{n}layers.json').read_text())
    moving,tools,extraction=internal_keepouts(meta,cache)
    tests.update(original_moving_or_internal_wire_overlap=S.overlap(patch,moving),
        internal_tool_corridor_overlap=S.overlap(delta,tools),
        internal_wire_extraction_envelope_overlap=S.overlap(delta,extraction),
        certified_motion_guard_overlap=S.overlap(new,S.load(OUT/f'sphere_{n}layers_motion_guard.npz')),
        original_fixed_outer_overlap=S.overlap(patch,S.load(cache/meta['free_layer_meshes'][0])))
    for path in S.fixed_environment_paths(n):tests['original_holder_context_'+path.stem]=S.overlap(patch,S.load(path))
    for other in report['supports']:
        if other['support_index']==index:continue
        amount=S.overlap(patch,S.load(OUT/other['mesh_file']))
        if amount>EPS:tests['other_sphere_support_'+str(other['support_index'])]=amount
    for name,value in tests.items():
        if value>1e-10:raise RuntimeError(f'Collar guard failed {n}/{index}: {name}={value}')
    poses=[]
    # Local assembled crop is centered at the actual carrier joint and includes
    # the complete new collar. Any original-body context remains in the crop.
    crop=S.profile(center,d,[(-.035,radius+.015),(.035,radius+.015)])
    for placement,sphere in enumerate(SPHERES):
        if sphere['layers']!=n:continue
        transform=sphere_transform(sphere);worldpatch=patch.transform(transform)
        if holder_full is not None:
            overlap=S.overlap(worldpatch,holder_full)
            if overlap>EPS:raise RuntimeError(f'New collar bonds holder support stock: {overlap}')
        row=dict(placement=placement,azimuth_deg=sphere['theta_deg'],old_part=strict(full.transform(transform)),new_part=strict(new.transform(transform)))
        if context is not None:
            local=context^crop.transform(transform)
            row['old_assembled_crop']=strict(local)
            row['new_assembled_crop']=strict(local+worldpatch)
        if not row['new_part']['passed'] or (context is not None and not row['new_assembled_crop']['passed']):
            raise RuntimeError('Repaired actual part/crop fails strict export: '+json.dumps(row))
        poses.append(row)
    if any(sha(ROOT/path)!=h for path,h in hashes.items()):raise RuntimeError('A frozen neck/stump file changed during repair checks')
    proof=dict(passed=True,layer_count=n,support_index=index,style='tapered positive-overlap removable root-carrier collar',
        root_carrier_center_mm=center.tolist(),axis=d.tolist(),stations_relative_to_carrier_mm=stations,
        first_section_outside_old_carrier_mm3=first_outside,convex_hull_fill_mm3=convex_error,seating_attempts=seating_attempts,service_corridor_halfspace_clip=service_clip,
        initial_vertex_checks=initial_vertex_checks,prefix_volume_threshold_mm3=EPS,prefix_checks=prefix_checks,
        all_height_prefix_reason='The collar is convex and its lowest vertices have finite interior balls contained by the earlier carrier. Its connected horizontal prefixes therefore retain that seated initial material as height increases.',
        guard_overlap_volumes_mm3=tests,original_full_sha256=sha(OUT/rec['mesh_file']),preserved_neck_stump_hashes=hashes,
        added_material_mm3=S.vol(delta),poses=poses,repair_source_sha256=sha(__file__))
    return patch,new,proof


def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def apply_prepared(prepared,inputs):
    """Commit only verified temporary stock, leaving every existing stump byte alone."""
    run_id='root_carrier_overlap_'+str(time.time_ns())
    stage=DIAG/run_id;stage.mkdir()
    backup=stage/'before';backup.mkdir()
    manifest_path=ROOT/'reports/custom_sphere_supports_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    reports={n:json.loads((OUT/f'sphere_{n}layers_supports.json').read_text()) for n in IDS}
    frozen={}
    for n,report in reports.items():
        if not report['passed'] or not report['finalization_complete']:raise RuntimeError('Input module is not finalized')
        for rec in report['supports']:
            for key in ['stump_mesh_file','conservative_retained_mesh_file']:
                path=OUT/rec[key];frozen[str(path.relative_to(ROOT))]=sha(path)
            if rec['kind']=='removable_fixed_support':
                for j in range(2):
                    path=OUT/f'sphere_{n}layers_neck_{rec["support_index"]:03d}_{j}.npz';frozen[str(path.relative_to(ROOT))]=sha(path)
        for suffix in ['retained_stumps','motion_envelope','motion_guard']+([] if n in RETAINED_ONLY_IDS else ['micro_reinforcements']):
            path=OUT/f'sphere_{n}layers_{suffix}.npz';frozen[str(path.relative_to(ROOT))]=sha(path)
    changed=[];proofs=[]
    for n,index,patch,new,proof in prepared:
        report=reports[n];rec=next(x for x in report['supports'] if x['support_index']==index)
        if rec.get('removable_body_repairs'):raise RuntimeError('Already repaired: refusing a repeated design change')
        oldfull=S.load(OUT/rec['mesh_file']);st=S.load(OUT/rec['stump_mesh_file'])
        old_removable=S.load(OUT/rec['removable_mesh_file'])
        baseline_partition_difference=S.vol(old_removable-(oldfull-st))+S.vol((oldfull-st)-old_removable)
        # Retain the accepted serialized stock partition and add only the
        # verified collar. Some historical raw Boolean graphs contain a tiny
        # reload discrepancy; do not silently change that baseline partition.
        removable=old_removable+patch
        proxy_name=f'sphere_{n}layers_removable_convex_{index:03d}_root_collar.npz'
        proxy_list=rec['generated_convex_sweep_meshes']+[proxy_name]
        cover=S.join([S.load(OUT/name) for name in rec['generated_convex_sweep_meshes']]+[patch])
        missing=S.vol(removable-cover);extra=S.vol(cover-removable)
        raw_missing=S.vol((new-st)-cover)
        if max(missing,raw_missing)>1e-8:raise RuntimeError(f'New proxy cover misses actual stock {n}/{index}: {missing}/{raw_missing}')
        old_decomposition=rec['removable_convex_decomposition']
        rec['generated_convex_sweep_meshes']=proxy_list
        rec['removable_convex_decomposition']=dict(passed=True,piece_count=len(proxy_list),
            conservative_sweep_outer_cover=True,final_saved_meshes_revalidated=True,
            maximum_convex_hull_fill_mm3=max(old_decomposition.get('maximum_convex_hull_fill_mm3',0.),proof['convex_hull_fill_mm3']),
            union_uncovered_actual_volume_mm3=missing,union_extra_proxy_volume_mm3=extra,
            union_uncovered_raw_full_minus_stump_volume_mm3=raw_missing,
            union_symmetric_difference_mm3=missing+extra,coverage_volume_tolerance_mm3=1e-8,
            source='Previously validated local convex sweep proxies plus the convex positive-overlap root-carrier collar. Their union conservatively covers the exact new nominal removable stock; reported extra proxy fill is not printed.')
        for filename,mesh in [(rec['mesh_file'],new),(rec['removable_mesh_file'],removable),(proxy_name,patch)]:
            S.save(stage/filename,mesh);changed.append((stage/filename,OUT/filename))
        saved_new=S.load(stage/rec['mesh_file']);saved_rem=S.load(stage/rec['removable_mesh_file'])
        stock_difference=S.vol(saved_rem-(saved_new-st))+S.vol((saved_new-st)-saved_rem)
        incremental_difference=S.vol(saved_rem-removable)+S.vol(removable-saved_rem)
        maximum=S.load(OUT/rec['conservative_retained_mesh_file'])
        retained_outside_maximum=S.vol((saved_new-saved_rem)-maximum)
        removable_outside_printed_full=S.vol(saved_rem-saved_new)
        if incremental_difference>1e-10 or max(retained_outside_maximum,removable_outside_printed_full)>1e-8:
            raise RuntimeError(f'Serialized stock containment failed {n}/{index}: remainder outside max={retained_outside_maximum}, removable outside printed={removable_outside_printed_full}, incremental={incremental_difference}')
        proof['preflight_inputs']=proof.get('inputs')
        proof.update(new_full_sha256=sha(stage/rec['mesh_file']),new_removable_sha256=sha(stage/rec['removable_mesh_file']),
            collar_proxy_mesh_file=proxy_name,collar_sha256=sha(stage/proxy_name),
            nominal_removable_partition_symmetric_difference_mm3=stock_difference,
            prior_nominal_removable_partition_symmetric_difference_mm3=baseline_partition_difference,
            accepted_removable_plus_collar_reload_symmetric_difference_mm3=incremental_difference,
            serialized_full_minus_removable_outside_maximum_retained_mm3=retained_outside_maximum,
            serialized_removable_outside_printed_full_mm3=removable_outside_printed_full,
            actual_stock_containment_tolerance_mm3=1e-8,
            nominal_state_ambiguity_mm3=stock_difference,
            partition_scope='Accepted prior nominal removable stock plus wholly removable collar. Raw-Boolean nominal partition ambiguity is reported; the actual serialized full-minus-removable state is contained in the unchanged maximum retained guard. Convex proxies additionally cover raw new-full-minus-stump.',
            proxy_missing_actual_mm3=missing,proxy_missing_raw_full_minus_stump_mm3=raw_missing,proxy_extra_fill_mm3=extra,
            run_id=run_id,inputs=inputs,application_source_sha256=sha(__file__))
        proof_path=ROOT/f'reports/custom_sphere_supports_joint_repair_{n}_{index:03d}.json'
        write_json(proof_path,proof)
        rec['added_volume_outside_original_mm3']+=proof['added_material_mm3']
        rec['removable_body_repairs']=[dict(kind='positive_overlap_root_carrier_collar',
            proof_report=str(proof_path.relative_to(ROOT)),proof_sha256=sha(proof_path),
            old_full_sha256=proof['original_full_sha256'],new_full_sha256=proof['new_full_sha256'],
            collar_proxy_mesh_file=proxy_name,added_material_mm3=proof['added_material_mm3'],
            original_necks_nominal_and_maximum_stumps_byte_identical=True)]
        proofs.append(proof)
    # All added collars must remain independent from every other repaired stock.
    for a,(n,index,patch,new,proof) in enumerate(prepared):
        for nn,j,other_patch,other_new,other_proof in prepared[:a]:
            if nn==n and S.overlap(patch,other_new)>EPS:raise RuntimeError('Collars fuse different removable parts')
    reclassifications=[]
    for n,indices in RETAINED_ONLY_IDS.items():
        report=reports[n];cache=ROOT/'cache/spheres';meta=json.loads((cache/f'sphere_{n}layers.json').read_text())
        moving,tools,extraction=internal_keepouts(meta,cache);guard=S.load(OUT/f'sphere_{n}layers_motion_guard.npz')
        outer=S.load(cache/meta['free_layer_meshes'][0]);environments=[(path,S.load(path)) for path in S.fixed_environment_paths(n)]
        for index in indices:
            rec=next(x for x in report['supports'] if x['support_index']==index)
            full=S.load(OUT/rec['mesh_file']);rem=S.load(OUT/rec['removable_mesh_file']);nominal=S.load(OUT/rec['stump_mesh_file']);maximum=S.load(OUT/rec['conservative_retained_mesh_file'])
            if not rem.is_empty():raise RuntimeError('Only actually empty removable meshes can be reclassified')
            checks=dict(full_outside_nominal_retained_mm3=S.vol(full-nominal),full_outside_maximum_retained_mm3=S.vol(full-maximum),
                moving_or_internal_wire_overlap_mm3=S.overlap(full,moving),motion_guard_overlap_mm3=S.overlap(full,guard),
                internal_tool_overlap_mm3=S.overlap(full,tools),internal_wire_extraction_overlap_mm3=S.overlap(full,extraction))
            contacts=[dict(environment_path=str(path.relative_to(ROOT)),fixed_contact_mm3=S.overlap(full,outer+env)) for path,env in environments]
            if max(checks.values())>1e-9 or any(x['fixed_contact_mm3']<=EPS for x in contacts):raise RuntimeError('Retained-only classification guard failed')
            bb=np.asarray(full.bounding_box());dimensions=bb[3:]-bb[:3]
            provenance_keys=['style','anchor_point','contact_point','centerline','cut_points','cut_plane_normals','end_profiles',
                'neck_length_mm','neck_length_build_mm','neck_diameter_mm','body_diameter_mm','contact_cap_diameter_mm',
                'minimum_neck_minor_diameter_normal_to_axis_mm','polygon_segments','polygon_inradius_factor',
                'horizontal_neck_diameter_mm','horizontal_body_diameter_mm','retained_height_from_each_contact_mm',
                'conservative_retained_height_from_each_contact_mm','nominal_cut_height_from_each_contact_mm']
            rec['discarded_untrimmed_branch_provenance']={key:rec.pop(key) for key in provenance_keys if key in rec}
            rec.update(kind='fixed_micro_reinforcement',style='permanent_retained_contact_cap',
                removal_required=False,printed_bounds_mm=bb.tolist(),printed_dimensions_mm=dimensions.tolist(),
                printed_volume_mm3=S.vol(full),classification_reason='The saved printed stock is a small clipped contact cap wholly contained by both old nominal and maximum retained meshes. Its saved removable mesh is empty; the untrimmed construction branch metadata did not describe the printed cap.',
                retained_only_classification_checks=checks,actual_fixed_contacts=contacts,generated_convex_sweep_meshes=[])
            for check in report['retained_state_service_check']['support_checks']:
                if check['support_index']==index:check['kind']='fixed_micro_reinforcement'
            reclassifications.append(dict(layer_count=n,support_index=index,full_sha256=sha(OUT/rec['mesh_file']),
                full_geometry_changed=False,removable_mesh_empty=True,printed_dimensions_mm=dimensions.tolist(),printed_volume_mm3=S.vol(full),
                checks=checks,actual_fixed_contacts=contacts))
        report['removable_support_count']=sum(r['kind']=='removable_fixed_support' for r in report['supports'])
        report['fixed_micro_reinforcement_count']=sum(r['kind']=='fixed_micro_reinforcement' for r in report['supports'])
        micro_name=f'sphere_{n}layers_micro_reinforcements.npz'
        micro=S.join([S.load(OUT/r['mesh_file']) for r in report['supports'] if r['kind']=='fixed_micro_reinforcement'])
        if S.vol(S.load(OUT/micro_name)-micro)>1e-10:raise RuntimeError('Reclassified micro aggregate lost prior fixed stock')
        S.save(stage/micro_name,micro);changed.append((stage/micro_name,OUT/micro_name))
    for n,report in reports.items():
        actual=[S.load(stage/r['mesh_file'] if (stage/r['mesh_file']).exists() else OUT/r['mesh_file']) for r in report['supports']]
        aggregate=S.join(actual);old_aggregate=S.load(OUT/f'sphere_{n}layers_supports.npz')
        collars=S.join([x[2] for x in prepared if x[0]==n])
        expected=old_aggregate+collars
        difference=S.vol(aggregate-expected)+S.vol(expected-aggregate)
        removed=S.vol(old_aggregate-aggregate)
        if difference>1e-8 or removed>1e-10:raise RuntimeError(f'Aggregate mismatch L{n}: {difference}/{removed}')
        meta=json.loads((ROOT/f'cache/spheres/sphere_{n}layers.json').read_text())
        outer=S.load(ROOT/'cache/spheres'/meta['free_layer_meshes'][0])
        for filename,mesh in [(f'sphere_{n}layers_supports.npz',aggregate),(f'sphere_{n}layers_supported_outer.npz',outer+aggregate)]:
            S.save(stage/filename,mesh);changed.append((stage/filename,OUT/filename))
        report['post_generation_removable_body_repairs']=dict(passed=True,run_id=run_id,
            support_indices=IDS[n],source_path=str(Path(__file__).relative_to(ROOT)),source_sha256=sha(__file__),
            added_material_mm3=sum(x[4]['added_material_mm3'] for x in prepared if x[0]==n),
            original_necks_nominal_and_maximum_stumps_byte_identical=True,
            aggregate_expected_union_symmetric_difference_mm3=difference,
            old_aggregate_material_removed_mm3=removed,
            full_model_build_validation='Required separately on fresh final assembly')
        report['retained_only_classifications']=[x for x in reclassifications if x['layer_count']==n]
        if not report['retained_state_service_check']['passed']:raise RuntimeError('Original retained service gate not passed')
        report['retained_state_service_check']['preserved_after_removable_body_repair']=True
        report['finalization_complete']=True;report['passed']=True
        write_json(stage/f'sphere_{n}layers_supports.json',report)
    # Validate inputs immediately before commit and preserve a byte-exact history.
    for path,h in inputs.items():
        if sha(ROOT/path)!=h:raise RuntimeError('Repair context changed before commit: '+path)
    for path,h in frozen.items():
        if sha(ROOT/path)!=h:raise RuntimeError('Protected input changed: '+path)
    backup_paths={target for _,target in changed if target.exists()}|{OUT/f'sphere_{n}layers_supports.json' for n in IDS}|{ROOT/f'reports/custom_sphere_supports_{n}.json' for n in IDS}|{manifest_path}
    backup_manifest=[]
    for path in sorted(backup_paths):
        destination=backup/str(path.relative_to(ROOT));destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        backup_manifest.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),backup=str(destination.relative_to(ROOT))))
    write_json(stage/'before_manifest.json',backup_manifest)
    for n in IDS:
        pending=json.loads((OUT/f'sphere_{n}layers_supports.json').read_text());pending.update(passed=False,finalization_complete=False,removable_body_repair_in_progress=run_id)
        write_json(OUT/f'sphere_{n}layers_supports.json',pending);write_json(ROOT/f'reports/custom_sphere_supports_{n}.json',pending)
    pending_manifest=dict(manifest);pending_manifest['generation_and_local_service_checks_passed']=False;pending_manifest['removable_body_repair_in_progress']=run_id
    write_json(manifest_path,pending_manifest)
    for source,target in changed:
        temporary=target.with_name(target.name+'.repair.part');shutil.copyfile(source,temporary);temporary.replace(target)
    for n,report in reports.items():
        write_json(OUT/f'sphere_{n}layers_supports.json',report);write_json(ROOT/f'reports/custom_sphere_supports_{n}.json',report)
    for path,h in frozen.items():
        if sha(ROOT/path)!=h:raise RuntimeError('Protected input changed during commit: '+path)
    for module in manifest['modules']:
        n=module['layer_count'];report=reports[n]
        module['temporary_count']=report['removable_support_count'];module['fixed_micro_count']=report['fixed_micro_reinforcement_count']
        module['styles']={style:sum(r['style']==style for r in report['supports']) for style in sorted(set(r['style'] for r in report['supports']))}
        module['maximum_micro_added_volume_mm3']=max(r['added_volume_outside_original_mm3'] for r in report['supports'] if r['kind']=='fixed_micro_reinforcement')
        module['retained_contact_cap_count']=len(RETAINED_ONLY_IDS.get(n,[]))
        module['metadata_sha256']=sha(ROOT/module['metadata_path'])
        module['maximum_convex_proxy_extra_fill_mm3']=max((r.get('removable_convex_decomposition',{}).get('union_extra_proxy_volume_mm3',0) for r in report['supports']),default=0.)
        for data in module['files'].values():
            path=ROOT/data['path'];data.update(bytes=path.stat().st_size,sha256=sha(path))
        module['removable_root_join_repairs']=report['post_generation_removable_body_repairs']
    manifest['removable_body_repair_run_id']=run_id;manifest['generation_and_local_service_checks_passed']=True
    manifest['total_placed_temporary_supports']=3*sum(r['removable_support_count'] for r in reports.values())
    manifest['total_placed_fixed_micro_reinforcements']=3*sum(r['fixed_micro_reinforcement_count'] for r in reports.values())
    manifest['retained_contact_cap_scope']='Seven saved support meshes (21 placements) are small permanent retained contact caps, with empty removable stock. Their full and maximum-retained geometries were not changed; metadata and fixed-color aggregate were corrected.'
    write_json(manifest_path,manifest)
    final=dict(passed=True,run_id=run_id,repair_source_sha256=sha(__file__),changed_support_count=len(prepared),
        placed_changed_support_count=3*len(prepared),total_added_module_material_mm3=sum(p['added_material_mm3'] for p in proofs),
        protected_file_hashes=frozen,protected_file_count=len(frozen),all_protected_files_byte_identical=True,
        input_hashes=inputs,old_input_backup_manifest=str((stage/'before_manifest.json').relative_to(ROOT)),
        retained_only_reclassifications=reclassifications,
        modules=[dict(layer_count=n,supports_path=f'cache/custom_sphere_supports/sphere_{n}layers_supports.npz',
            supports_sha256=sha(OUT/f'sphere_{n}layers_supports.npz'),metadata_sha256=sha(OUT/f'sphere_{n}layers_supports.json')) for n in IDS],
        full_model_build_and_external_removal_validation='Required separately after the fresh complete model assembly')
    write_json(ROOT/'reports/custom_sphere_supports_joint_repairs.json',final)
    print(json.dumps(final,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layers',type=int,default=3);parser.add_argument('--index',type=int,default=15)
    parser.add_argument('--all',action='store_true');parser.add_argument('--apply',action='store_true');parser.add_argument('--from-saved',action='store_true');args=parser.parse_args()
    if args.apply and not args.all:raise RuntimeError('Coherent application requires --all')
    DIAG.mkdir(exist_ok=True);started=time.monotonic()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'cache/fresh_v6_float64_trial.npz',ROOT/'cache/custom_holder_supports/holder_supports.npz']}
    context=None if args.from_saved else S.load(ROOT/'cache/fresh_v6_float64_trial.npz');holder=S.load(ROOT/'cache/custom_holder_supports/holder_supports.npz')
    pairs=[(n,index) for n,indices in IDS.items() for index in indices] if args.all else [(args.layers,args.index)]
    prepared=[]
    for n,index in pairs:
        stem=f'sphere_{n}layers_support_{index:03d}_root_join'
        if args.from_saved:
            proof=json.loads((ROOT/f'reports/custom_sphere_supports_joint_repair_{n}_{index:03d}.json').read_text())
            if not proof['passed']:raise RuntimeError('Saved candidate failed')
            rec=next(x for x in json.loads((OUT/f'sphere_{n}layers_supports.json').read_text())['supports'] if x['support_index']==index)
            if sha(OUT/rec['mesh_file'])!=proof['original_full_sha256']:raise RuntimeError('Saved candidate original full changed')
            for path,h in proof['preserved_neck_stump_hashes'].items():
                if sha(ROOT/path)!=h:raise RuntimeError('Saved candidate protected input changed')
            context_path='cache/fresh_v6_float64_trial.npz'
            if proof['inputs'][context_path]!=inputs[context_path]:raise RuntimeError('Saved assembled-crop context changed')
            patch=S.load(DIAG/(stem+'_collar.npz'));new=S.load(DIAG/(stem+'_full.npz'))
            expected=S.load(OUT/rec['mesh_file'])+patch
            if S.vol(new-expected)+S.vol(expected-new)>1e-10:raise RuntimeError('Saved new full differs from old full plus exact collar')
            checks=[]
            for placement,sphere in enumerate(SPHERES):
                if sphere['layers']!=n:continue
                amount=S.overlap(patch.transform(sphere_transform(sphere)),holder)
                checks.append(dict(placement=placement,overlap_mm3=amount))
                if amount>EPS:raise RuntimeError('Saved collar now collides with current holder stock')
            proof['current_holder_guard_revalidation']=dict(passed=True,sha256=inputs['cache/custom_holder_supports/holder_supports.npz'],checks=checks)
        else:patch,new,proof=prepare(n,index,context,holder)
        prepared.append((n,index,patch,new,proof))
        S.save(DIAG/(stem+'_collar.npz'),patch);S.save(DIAG/(stem+'_full.npz'),new)
        proof['elapsed_seconds']=time.monotonic()-started
        if not args.from_saved:proof['inputs']=inputs
        write_json(ROOT/f'reports/custom_sphere_supports_joint_repair_{n}_{index:03d}.json',proof)
        print(json.dumps(dict(layer_count=n,support_index=index,passed=True,added_material_mm3=proof['added_material_mm3'],
            poses=[dict(placement=x['placement'],old_part=x['old_part']['passed'],new_part=x['new_part']['passed'],old_crop=x['old_assembled_crop']['passed'],new_crop=x['new_assembled_crop']['passed']) for x in proof['poses']])),flush=True)
    if args.apply:apply_prepared(prepared,inputs)
    print('Completed in',time.monotonic()-started,'seconds',flush=True)


if __name__=='__main__':main()
