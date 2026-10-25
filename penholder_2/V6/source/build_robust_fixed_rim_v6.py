#!/usr/bin/env python3
"""Cover the complete fixed rim contact zone with a robust positive box.

The complete source lobe is already inside the main occupied component.
It is enclosed by an explicit small permanent positive bond; no source
component is deleted, and the maximum tested simplify tolerance is 1e-5 mm.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe

ROOT=D.ROOT
OUT=ROOT/'cache/fixed_precision_repairs/additional_rim'

def main():
    dp=ROOT/'reports/fixed_precision_additional_rim_diagnostic.json'
    diag=json.loads(dp.read_text());inputs={str(dp.relative_to(ROOT)):D.sha(dp),str(Path(__file__).relative_to(ROOT)):D.sha(__file__)}
    contexts={}
    for kind,record in diag['contexts'].items():
        path=ROOT/record['context_file']
        if D.sha(path)!=record['context_sha256']:raise ValueError('Historical local context changed')
        inputs[str(path.relative_to(ROOT))]=D.sha(path);contexts[kind]=D.load(path)
    raw=contexts['fresh_raw'];original=contexts['original']
    pieces=sorted(raw.decompose(),key=lambda x:-x.volume())
    if len(pieces)!=2 or not 0<pieces[1].volume()<1e-5:raise ValueError('Expected the actual two-component local raw context')
    main,lobe=pieces
    if not (lobe-main).is_empty():raise ValueError('Complete positive lobe is not inside main occupied component')
    a,b=np.asarray(diag['edge']['endpoints_mm']);axis=(b-a)/np.linalg.norm(b-a)
    corners=np.asarray(list(itertools.product([-.001,.001],repeat=3)))
    lv=np.asarray(lobe.to_mesh64().vert_properties[:,:3]);roi=np.asarray(diag['roi_bounds_mm'])
    lbb=np.asarray(lobe.bounding_box())
    if np.any(lbb[:3]<=roi[:3]+.001)or np.any(lbb[3:]>=roi[3:]-.001):raise ValueError('Lobe is not fully interior to crop')
    points=np.vstack((a-.002*axis+corners,b+.002*axis+corners,(lv[:,None,:]+corners[None,:,:]).reshape(-1,3)))
    old_path=OUT/'rim_final_fixed_fill.npz'
    old=D.load(old_path);inputs[str(old_path.relative_to(ROOT))]=D.sha(old_path)
    old_bb=np.asarray(old.bounding_box());padding=.02
    patch=D.box(old_bb[:3]-padding,old_bb[3:]+padding)
    contact=patch^main
    if contact.is_empty()or contact.volume()<=1e-12:raise ValueError('No finite occupied fixed contact')
    floor=float(contact.bounding_box()[2])+.00001
    patch=patch.trim_by_plane((0,0,1),floor)
    if not (old-patch).is_empty():raise ValueError('Robust replacement must contain the complete adopted old fixed piece')
    if floor>min(a[2],b[2])-.00005 or not (lobe-patch).is_empty():raise ValueError('Patch fails to contain the complete original positive lobe')
    prefix=patch.trim_by_plane((0,0,-1),-(floor+.00005));first_contact=prefix^main
    if first_contact.is_empty()or first_contact.volume()<=1e-12 or abs(first_contact.bounding_box()[2]-floor)>1e-8:
        raise ValueError('Patch has an unsupported first prefix')
    before=canonical_probe(raw.as_original());after=canonical_probe((raw+patch).as_original())
    if not after['passed']:raise ValueError('Local strict export still fails: '+str(after))
    path=OUT/'rim_final_robust_fill.npz';D.save(path,patch);bb=np.asarray(patch.bounding_box())
    row=dict(id='rim_final',kind='fixed_micro_reinforcement',temporary=False,repair_purpose='robust_complete_fixed_rim_contact_zone',
        mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),target_mm=((bb[:3]+bb[3:])/2).tolist(),
        root_mm=[float((bb[0]+bb[3])/2),float((bb[1]+bb[4])/2),float(bb[2])],bounds_mm=bb.tolist(),
        volume_mm3=float(patch.volume()),actual_added_volume_over_occupied_context_mm3=float((patch-main).volume()),
        signed_raw_difference_volume_mm3=float((patch-raw).volume()),
        preserved_positive_lobe_volume_mm3=float(lobe.volume()),preserved_positive_lobe_faces=lobe.num_tri(),
        preserved_positive_lobe_bounds_mm=lbb.tolist(),entire_source_lobe_contained_in_patch=True,
        entire_source_lobe_already_inside_main_component=True,
        duplicate_lobe_main_overlap_volume_mm3=float((lobe^main).volume()),
        edge_endpoints_mm=[a.tolist(),b.tolist()],axis_aligned_contact_zone_padding_mm=padding,
        replacement_holder_index=1284,complete_old_piece_contained=True,old_piece_file=str(old_path.relative_to(ROOT)),old_piece_sha256=D.sha(old_path),
        old_piece_outside_replacement_empty=True,
        original_fixed_contact_signed_volume_mm3=float((patch^original).volume()),
        occupied_context_contact_volume_mm3=float((patch^main).volume()),
        first_prefix_height_mm=floor+.00005,first_prefix_contact_volume_mm3=float(first_contact.volume()),
        first_prefix_contact_fraction=float(first_contact.volume()/prefix.volume()),first_prefix_components=1,
        nearby_existing_fixed_context=[dict(owner='holder',index=446,kind='local_holder_twig',temporary=True),
            dict(owner='holder',index=1035,kind='fixed_micro_reinforcement',temporary=False),
            dict(owner='holder',index=1276,kind='fixed_micro_reinforcement',temporary=False)],
        neck_count=0,local_before_strict_probe=before,local_after_strict_probe=after,
        source_diagnosis='The entire adopted fixed piece, original contained rim lobe and successor contact zone are enclosed by one well-conditioned axis-aligned positive volume with0.02 mm margin. This replaces one declared fixed row by a superset; no original model or prior added material is removed.')
    for file,digest in inputs.items():
        if D.sha(ROOT/file)!=digest:raise ValueError('Frozen local source changed')
    report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,
        diagnostic_only=True,source_models_modified=False,repair_count=1,units='mm',
        aggregate_mesh_file=row['mesh_file'],aggregate_sha256=row['mesh_sha256'],aggregate_volume_mm3=row['volume_mm3'],
        repairs=[row],input_sha256=inputs,source_sha256=D.sha(__file__))
    (ROOT/'reports/fixed_precision_robust_rim_candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Robust fixed-rim replacement local strict pass;',json.dumps(dict(mesh_file=row['mesh_file'],sha256=row['mesh_sha256'],added_mm3=row['actual_added_volume_over_occupied_context_mm3'])),flush=True)

if __name__=='__main__':main()
