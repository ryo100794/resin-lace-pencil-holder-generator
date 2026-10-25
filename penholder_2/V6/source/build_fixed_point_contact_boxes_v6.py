#!/usr/bin/env python3
"""Replace the two measured fixed-rim point contacts by bounded positive boxes.

These are deliberate declared fixed-geometry additions. Each new piece is
a superset of its old retained fill. No source or adopted cache is edited.
"""
from pathlib import Path
import json
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe

ROOT=D.ROOT
OUT=ROOT/'cache/fixed_precision_repairs/robust_point_boxes'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_path=ROOT/'reports/fixed_precision_repair_candidates.json'
    source=json.loads(source_path.read_text());inputs={str(source_path.relative_to(ROOT)):D.sha(source_path),str(Path(__file__).relative_to(ROOT)):D.sha(__file__)}
    rows=[];solids=[]
    for ident,index in [('rim_20',1274),('rim_21',1275)]:
        old_record=next(x for x in source['repairs']if x['id']==ident)
        old_path=ROOT/old_record['mesh_file'];old=D.load(old_path);inputs[str(old_path.relative_to(ROOT))]=D.sha(old_path)
        adopted=ROOT/f'cache/custom_holder_supports/holder_support_{index:04d}_full.npz'
        if D.sha(adopted)!=D.sha(old_path):raise ValueError('Expected the exact adopted original fixed fill')
        raw_path=ROOT/f'cache/fixed_precision_repairs/{ident}_fresh_raw_context.npz'
        original_path=ROOT/f'cache/fixed_precision_repairs/{ident}_original_context.npz'
        raw=D.load(raw_path);original=D.load(original_path)
        for path in [raw_path,original_path]:inputs[str(path.relative_to(ROOT))]=D.sha(path)
        parts=sorted(raw.decompose(),key=lambda x:-x.volume())
        if len(parts)!=2:raise ValueError('Unexpected original rim context topology')
        main,lobe=parts
        if not (lobe-main).is_empty():raise ValueError('Original positive rim lobe is not fully contained')
        bb=np.asarray(old.bounding_box());padding=.02
        replacement=D.box(bb[:3]-padding,bb[3:]+padding)
        contact=replacement^main
        if contact.is_empty()or contact.volume()<=1e-12:raise ValueError('No positive fixed contact')
        floor=float(contact.bounding_box()[2])+.00001
        replacement=replacement.trim_by_plane((0,0,1),floor)
        if not (old-replacement).is_empty()or not(lobe-replacement).is_empty():raise ValueError('Replacement is not a complete superset')
        prefix=replacement.trim_by_plane((0,0,-1),-(floor+.00005));first=prefix^main
        if first.is_empty()or first.volume()<=1e-12 or abs(first.bounding_box()[2]-floor)>1e-8:raise ValueError('Replacement first prefix is unseated')
        before=canonical_probe(raw.as_original());after=canonical_probe((raw+replacement).as_original())
        if not after['passed']:raise ValueError('Original context strict export failed after replacement')
        path=OUT/f'{ident}_fixed_box.npz';D.save(path,replacement);solids.append(replacement)
        bounds=np.asarray(replacement.bounding_box())
        rows.append(dict(id=ident,kind='fixed_micro_reinforcement',temporary=False,
            replacement_holder_index=index,repair_purpose='robust_fixed_rim_point_contact_zone',
            mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),volume_mm3=float(replacement.volume()),
            actual_added_volume_over_occupied_context_mm3=float((replacement-main).volume()),
            complete_old_piece_contained=True,old_piece_outside_replacement_empty=True,
            old_piece_file=str(old_path.relative_to(ROOT)),old_piece_sha256=D.sha(old_path),
            historical_adopted_full_mesh_file=str(adopted.relative_to(ROOT)),historical_adopted_sha256=D.sha(adopted),
            axis_aligned_contact_zone_padding_mm=padding,
            target_mm=((bounds[:3]+bounds[3:])/2).tolist(),root_mm=[float((bounds[0]+bounds[3])/2),float((bounds[1]+bounds[4])/2),floor],
            bounds_mm=bounds.tolist(),neck_count=0,
            first_prefix_height_mm=floor+.00005,first_prefix_contact_volume_mm3=float(first.volume()),
            first_prefix_contact_fraction=float(first.volume()/prefix.volume()),first_prefix_components=1,
            preserved_positive_lobe_volume_mm3=float(lobe.volume()),
            entire_source_lobe_contained_in_patch=True,entire_source_lobe_already_inside_main_component=True,
            occupied_context_contact_volume_mm3=float((replacement^main).volume()),
            nearby_existing_fixed_context=old_record['nearby_existing_fixed_context'],
            local_before_strict_probe=before,local_after_strict_probe=after,
            source_diagnosis='Two separate float64 surface fans already coincide at the same fixed contact point. A well-conditioned positive box encloses the entire old fixed contact fill with0.02 mm margin; this is an explicit geometry addition, not a rounding perturbation or relaxed validator.'))
        print(ident,'replacement',index,'local strict passed',rows[-1]['actual_added_volume_over_occupied_context_mm3'],flush=True)
    aggregate=mf.Manifold.batch_boolean(solids,mf.OpType.Add);path=OUT/'fixed_point_contact_boxes.npz';D.save(path,aggregate)
    for path,digest in inputs.items():
        if D.sha(ROOT/path)!=digest:raise ValueError('Historical source changed')
    report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,
        diagnostic_only=True,source_models_modified=False,repair_count=2,units='mm',
        replacements_only_no_extra_part_count=True,aggregate_mesh_file=str((OUT/'fixed_point_contact_boxes.npz').relative_to(ROOT)),
        aggregate_sha256=D.sha(OUT/'fixed_point_contact_boxes.npz'),aggregate_volume_mm3=float(aggregate.volume()),
        repairs=rows,input_sha256=inputs,source_sha256=D.sha(__file__))
    (ROOT/'reports/fixed_precision_point_boxes_candidates.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
