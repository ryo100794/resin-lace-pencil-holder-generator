#!/usr/bin/env python3
"""Repair only H523/H529 coincident carrier rings with removable collars.

The untouched V5 source and every other holder part are immutable inputs.
The added collar begins inside the constant-radius carrier, surrounds the
coincident terminal ring with positive volume, and lies entirely beyond both
endpoint cuts.  Final whole-model export/removal validation remains separate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

import custom_holder_supports as H
import build_test_coupon as B

ROOT=H.ROOT
OUT=H.OUT
IDS=(523,529)
TRIAL=OUT/'carrier_joint_repair_trials'
REPORT=OUT/'carrier_joint_repairs.json'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def canonical(solid):
    try:
        _,_,report=B.canonical_float32_mesh(solid.simplify(1e-5))
        return dict(passed=True,report=report)
    except Exception as error:
        trace=error.__traceback__;frame={}
        while trace:
            if trace.tb_frame.f_code.co_name=='canonical_float32_mesh':
                frame=trace.tb_frame.f_locals;break
            trace=trace.tb_next
        return dict(passed=False,error=str(error),bad_edges=frame.get('bad_edges'),
                    near_zero=frame.get('near_zero'))


def bounds_overlap(a,b):
    return bool(np.all(a[:3]<=b[3:]) and np.all(b[:3]<=a[3:]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();started=time.monotonic();TRIAL.mkdir(exist_ok=True)
    report_paths=[OUT/'holder_supports.json',ROOT/'reports/custom_holder_supports.json',OUT/'generation_checkpoint.json']
    documents=[json.loads(path.read_text()) for path in report_paths]
    manifest=documents[0];rows=manifest['parts']
    require(len(rows)==1274,'Unexpected holder part count')
    require(not any(rows[i].get('carrier_joint_repair') for i in IDS),'Repair has already been applied')
    old_aggregate=OUT/'holder_supports.npz'
    source=ROOT/'cache/original_v5.npz'
    protected=[source,old_aggregate,OUT/'holder_retained_stumps.npz',OUT/'holder_micro_reinforcements.npz',*report_paths]
    input_hashes={str(path.relative_to(ROOT)):H.sha(path) for path in protected}
    all_full_hashes={r['full_mesh_file']:H.sha(OUT/r['full_mesh_file']) for r in rows}
    before=TRIAL/'before';before.mkdir(exist_ok=True)
    stage=TRIAL/'stage';stage.mkdir(exist_ok=True)
    generated={};repairs=[]
    for index in IDS:
        row=rows[index]
        require(row['index']==index and row['support_style']=='pitched_carriers','Unexpected source part')
        require(row['removal_mode']=='whole_piece' and len(row['removal_fragments'])==1,'Unexpected saved cut plan')
        full,_,_=H.load(OUT/row['full_mesh_file']);removable,_,_=H.load(OUT/row['removable_mesh_file'])
        retained,_,_=H.load(OUT/row['retained_mesh_file']);fixed=full-removable
        q=np.asarray(row['root_mm']);nr=np.asarray(row['neck_axes'][0]);rb=row['body_diameter_mm']/2;rn=row['neck_diameter_mm']/2
        nz=float(nr[2]);cr=max(.60,(rb+rn*math.sqrt(1-nz*nz)+.22*nz+.045)/nz)
        require(cr-.025>.34,'Initial collar ring is outside the constant-radius carrier')
        stations=[(cr-.025,rb-.003),(cr-.010,rb+.001),(cr+.010,rb+.001),(cr+.020,rb-.003)]
        collar=H.axial_profile(q,nr,stations);added=collar-full
        new_full=full+collar;new_removable=removable+collar
        first=H.low_axis(collar,q,nr,cr-.025+.0001)
        seating=H.volume(first-full);retained_overlap=H.volume(collar^retained);fixed_overlap=H.volume(collar^fixed)
        fixed_after=new_full-new_removable
        fixed_difference=H.volume(fixed_after-fixed)+H.volume(fixed-fixed_after)
        necks=[H.axial_profile(point,normal,[(.08,rn),(.22,rn)]) for point,normal in zip([q,np.asarray(row['target_mm'])],row['neck_axes'])]
        neck_overlap=sum(H.volume(collar^neck) for neck in necks)
        prefix_passed,prefix=H.add_prefix_checks(collar,full,float(collar.bounding_box()[5]))
        old_topology=canonical(full);new_topology=canonical(new_full)
        require(not old_topology['passed'] and old_topology.get('bad_edges')==2,'Expected carrier seam was not reproduced')
        require(new_topology['passed'],'Collared part failed the unchanged strict export gate')
        require(max(seating,retained_overlap,fixed_overlap,fixed_difference,neck_overlap)<=1e-12,'Collar altered a seat, waist or retained state')
        require(prefix_passed,'Added collar has an unsupported prefix')
        env=json.loads((ROOT/'reports/moving_envelopes.json').read_text())
        motion_passed,motion_clearance=H.sphere_envelope_ok(collar,env)
        require(motion_passed,'Collar enters original conservative moving envelope')
        convex_names=list(row['generated_convex_sweep_meshes'])
        convex=[H.load(OUT/name)[0] for name in convex_names]
        convex.append(collar);coverage=H.union(convex)
        missing=H.volume(new_removable-coverage);extra=H.volume(coverage-new_removable)
        convex_fill=H.volume(collar.hull()-collar)
        require(max(missing,extra,convex_fill)<=1e-8,'Updated convex sweep stock does not cover the actual removable solid')
        new_name=f'holder_support_{index:04d}_convex_sweep_piece_{len(convex_names):02d}.npz'
        for name in [row['full_mesh_file'],row['removable_mesh_file'],row['removal_fragments'][0]['mesh_file']]:
            shutil.copy2(OUT/name,before/name)
        for name,obj in [(row['full_mesh_file'],new_full),(row['removable_mesh_file'],new_removable),
                         (row['removal_fragments'][0]['mesh_file'],new_removable),(new_name,collar)]:
            H.save(stage/name,obj)
        generated[index]=dict(old_full=full,collar=collar,added=added,new_full=new_full,new_removable=new_removable,
                              new_convex_name=new_name,source_row=row)
        repair=dict(support_index=index,source_full_mesh_sha256=all_full_hashes[row['full_mesh_file']],
            source_full_snapshot=str((before/row['full_mesh_file']).relative_to(ROOT)),
            source_removable_snapshot=str((before/row['removable_mesh_file']).relative_to(ROOT)),
            purpose='Replace a coincident carrier terminal seam with positive volumetric overlap entirely inside temporary removable stock.',
            carrier_endpoint='root',carrier_terminal_axis_offset_mm=cr,collar_axis=nr.tolist(),
            collar_stations_axis_offset_and_radius_mm=[[float(s),float(r)] for s,r in stations],
            maximum_collar_diameter_mm=2*(rb+.001),collar_volume_mm3=H.volume(collar),
            material_added_outside_previous_support_mm3=H.volume(added),
            initial_axial_0_0001_mm_unseated_volume_mm3=seating,
            existing_neck_overlap_mm3=neck_overlap,maximum_retained_overlap_mm3=retained_overlap,
            nominal_fixed_overlap_mm3=fixed_overlap,nominal_fixed_state_symmetric_difference_mm3=fixed_difference,
            collar_prefix_connectivity_to_existing_stock_passed=prefix_passed,collar_prefix_checks=prefix,
            conservative_moving_envelope_check_passed=motion_passed,moving_envelope_clearance_mm=motion_clearance,
            original_individual_strict_export=old_topology,repaired_individual_strict_export=new_topology,
            convex_sweep_coverage_missing_mm3=missing,convex_sweep_coverage_extra_mm3=extra,collar_convex_hull_extra_mm3=convex_fill,
            neck_dimensions_and_both_cut_planes_preserved=True,nominal_and_maximum_retained_files_preserved=True)
        repairs.append(repair)
        print('Strict local repair passed',index,flush=True)

    # Only this small additive delta requires new context checks. The earlier
    # full support paths, cut locations and contact geometry are unchanged.
    original=H.load(source)[0];holder=H.load(old_aggregate)[0]
    for repair in repairs:
        index=repair['support_index'];entry=generated[index];row=entry['source_row']
        added=entry['added'];collar=entry['collar'];new_removable=entry['new_removable']
        original_overlap=H.volume(added^original);other_overlap=H.volume(added^holder)
        removable_original_overlap=H.volume(new_removable^original)
        fragment=row['removal_fragments'][0]
        query=H.sweep(collar,fragment['direction'],fragment['travel_mm'])
        swept_original_overlap=H.volume(query^original)
        previous_overlap=0.;affected_previous=[]
        qb=np.asarray(query.bounding_box())
        for previous in rows:
            if previous['index']==index:continue
            if previous['index']>index and previous.get('temporary',True):continue
            endpoints=np.asarray([previous['root_mm'],previous['target_mm']])
            conservative_bounds=np.r_[endpoints.min(axis=0)-2.,endpoints.max(axis=0)+2.]
            if not bounds_overlap(qb,conservative_bounds):continue
            solid=H.load(OUT/previous['full_mesh_file'])[0]
            if not H.bbox_intersects(query,solid):continue
            overlap=H.volume(query^solid);previous_overlap+=overlap
            if overlap>1e-10:affected_previous.append(dict(index=previous['index'],overlap_mm3=overlap))
        require(max(original_overlap,other_overlap,removable_original_overlap,swept_original_overlap,previous_overlap)<=1e-8,
                'Collar introduced a model contact, support fusion or blocked generation withdrawal')
        repair.update(added_original_overlap_mm3=original_overlap,added_previous_holder_overlap_mm3=other_overlap,
            new_removable_original_overlap_mm3=removable_original_overlap,
            collar_swept_original_overlap_mm3=swept_original_overlap,
            collar_swept_earlier_or_permanent_holder_overlap_mm3=previous_overlap,
            affected_previous_supports=affected_previous,
            existing_withdrawal_direction=fragment['direction'],existing_withdrawal_distance_mm=fragment['travel_mm'],
            withdrawal_check_scope='Added collar versus original and every earlier holder support or permanent holder addition; the final all-support/all-stump/+Z-exit planner remains authoritative.',passed=True)
        print('Context and added-stock withdrawal passed',index,flush=True)
    del original,holder

    # Rebuild the aggregate from exactly the declared individual parts.
    final_parts=[generated[row['index']]['new_full'] if row['index'] in generated else H.load(OUT/row['full_mesh_file'])[0] for row in rows]
    aggregate=H.union(final_parts);H.save(stage/'holder_supports.npz',aggregate)
    new_aggregate_sha=H.sha(stage/'holder_supports.npz')
    unchanged_retained_sha=input_hashes[str((OUT/'holder_retained_stumps.npz').relative_to(ROOT))]
    summary=dict(passed=True,applied=args.apply,repair_source=str(Path(__file__).relative_to(ROOT)),repair_source_sha256=H.sha(__file__),
        modified_part_indices=list(IDS),part_count_before=len(rows),part_count_after=len(rows),
        original_V5_input_sha256=input_hashes[str(source.relative_to(ROOT))],input_sha256=input_hashes,
        material_added_mm3=sum(r['material_added_outside_previous_support_mm3'] for r in repairs),
        holder_supports_output_sha256=new_aggregate_sha,retained_aggregate_unchanged_sha256=unchanged_retained_sha,
        parts=repairs,other_1272_full_meshes_unchanged=True,retained_and_permanent_geometry_unchanged=True,
        whole_model_export_and_combined_removal_validation='Required separately after all agents finish their declared repairs.',
        elapsed_seconds=time.monotonic()-started)
    for document in documents:
        for repair in repairs:
            index=repair['support_index'];entry=generated[index];row=document['parts'][index]
            row['carrier_joint_repair']=copy.deepcopy(repair)
            row['generated_convex_sweep_meshes'].append(entry['new_convex_name'])
            row['removal_fragments'][0]['bounds_mm']=list(entry['new_removable'].bounding_box())
            row['maximum_local_body_diameter_mm']=repair['maximum_collar_diameter_mm']
        document.setdefault('post_generation_repairs',[]).append(dict(kind='positive_overlap_root_carrier_collars',
            modified_part_indices=list(IDS),report=str(REPORT.relative_to(ROOT)),source_sha256=H.sha(__file__),passed=True))
        if 'support_volume_mm3' in document:document['support_volume_mm3']=H.volume(aggregate)
    for path,document in zip(report_paths,documents):
        (stage/(path.name+'.document.json')).write_text(json.dumps(document,indent=2)+'\n')
    (TRIAL/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    for relative,digest in input_hashes.items():require(H.sha(ROOT/relative)==digest,'Input changed during repair: '+relative)
    for name,digest in all_full_hashes.items():require(H.sha(OUT/name)==digest,'A source part changed during repair: '+name)
    if args.apply:
        for index,entry in generated.items():
            row=entry['source_row']
            names=[row['full_mesh_file'],row['removable_mesh_file'],row['removal_fragments'][0]['mesh_file'],entry['new_convex_name']]
            for name in names:(stage/name).replace(OUT/name)
        (stage/'holder_supports.npz').replace(old_aggregate)
        for path,document in zip(report_paths,documents):
            temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(document,indent=2)+'\n');temporary.replace(path)
        REPORT.write_text(json.dumps(summary,indent=2)+'\n')
        for row in rows:
            if row['index'] not in IDS:require(H.sha(OUT/row['full_mesh_file'])==all_full_hashes[row['full_mesh_file']],'An unrelated part changed')
        require(H.sha(OUT/'holder_retained_stumps.npz')==unchanged_retained_sha,'Retained aggregate changed')
        require(H.sha(source)==input_hashes[str(source.relative_to(ROOT))],'Original V5 changed')
        print('APPLIED and frozen',new_aggregate_sha,flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['parts','input_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
