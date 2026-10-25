#!/usr/bin/env python3
"""Construct only the eight diagnosed fixed rim bonds and two cavity fills.

All additions are explicit retained fixed material. Source model/support
files are read-only. Global topology, material starts and removal gates are
still required after the pieces are incorporated into the complete assembly.
"""
from pathlib import Path
import itertools
import json
import time
import numpy as np
import manifold3d as mf
import build_test_coupon as B
import diagnose_fixed_precision_v6 as D

ROOT=D.ROOT
OUT=D.OUT/'candidates'

def canonical_probe(model):
    try:
        v,f,result=B.canonical_float32_mesh(model.simplify(1e-5))
        return dict(passed=True,vertices=len(v),triangles=len(f),components=result['components'])
    except Exception as error:
        row=dict(passed=False,exception=str(error));tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name=='canonical_float32_mesh':
                q=tb.tb_frame.f_locals
                for key in ['collapsed','zero_area','near_zero','bad_edges','error']:
                    if key in q:row[key]=int(q[key]) if key!='error' else float(q[key])
                if 'components' in q:row['component_volumes_mm3']=[float(c.volume())for c in q['components']]
            tb=tb.tb_next
        return row

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    input_path=ROOT/'reports/fixed_precision_source_attribution.json'
    scope=json.loads(input_path.read_text());rows=[];patches=[];input_hashes={str(input_path.relative_to(ROOT)):D.sha(input_path)}
    for case in scope['cases']:
        original_path=ROOT/case['original']['context_file'];raw_path=ROOT/case['fresh_raw']['context_file']
        for path in [original_path,raw_path]:input_hashes[str(path.relative_to(ROOT))]=D.sha(path)
        original=D.load(original_path);raw=D.load(raw_path)
        occupied_reference=raw
        before=canonical_probe(raw)
        if case['kind']=='original_rim_edge':
            a,b=np.asarray(case['edge']['endpoints_mm']);direction=b-a;direction/=np.linalg.norm(direction)
            half_width=.001;extension=.002
            corners=np.asarray(list(itertools.product([-half_width,half_width],repeat=3)))
            components=sorted(raw.decompose(),key=lambda q:-float(q.volume()))
            lobes=components[1:]
            roi=np.asarray(case['roi_bounds_mm'])
            if len(lobes)!=1 or not 0<float(lobes[0].volume())<1e-5:
                raise ValueError('Expected one measured positive rim seam lobe')
            lobe=lobes[0];lbb=np.asarray(lobe.bounding_box())
            if np.any(lbb[:3]<=roi[:3]+.001) or np.any(lbb[3:]>=roi[3:]-.001):
                raise ValueError('Scoped lobe touches a crop boundary; not an isolated source component')
            lv=np.asarray(lobe.to_mesh64().vert_properties[:,:3])
            # The diagnosed positive lobe is duplicate occupied material,
            # entirely inside the large context component. Raw signed
            # volumes therefore double-count it; do not clamp a negative
            # raw Boolean difference and call that actual added volume.
            occupied_reference=components[0]
            if not (lobe-occupied_reference).is_empty():
                raise ValueError('Rim lobe is not fully inside its large occupied context')
            points=np.vstack((a-extension*direction+corners,b+extension*direction+corners,
                              (lv[:,None,:]+corners[None,:,:]).reshape(-1,3)))
            patch=mf.Manifold.hull_points(points)
            contact=patch^raw
            if D.volume(contact)<=1e-12:raise ValueError('No finite rim contact')
            floor=float(contact.bounding_box()[2])+.00001
            patch=patch.trim_by_plane((0,0,1),floor)
            if floor>min(a[2],b[2])-.00005:raise ValueError('Root trim would expose diagnosed edge')
            if not (lobe-patch).is_empty():raise ValueError('Positive original seam lobe was not fully retained')
            detail=dict(edge_index=case['edge']['index'],edge_endpoints_mm=[a.tolist(),b.tolist()],
                        bounding_cube_half_width_mm=half_width,endpoint_extension_mm=extension,
                        preserved_positive_lobe_volume_mm3=float(lobe.volume()),
                        preserved_positive_lobe_bounds_mm=lbb.tolist(),
                        preserved_positive_lobe_faces=lobe.num_tri(),
                        entire_source_lobe_contained_in_patch=True,
                        entire_source_lobe_already_inside_main_component=True,
                        duplicate_lobe_main_overlap_volume_mm3=float((lobe^occupied_reference).volume()),
                        source_diagnosis='The exact nonmanifold edge lies on the immutable V5 fixed rim.')
        else:
            void_path=ROOT/case['positive_void_file'];input_hashes[str(void_path.relative_to(ROOT))]=D.sha(void_path)
            void=D.load(void_path);bb=np.asarray(void.bounding_box());padding=.0005
            patch=D.box(bb[:3]-padding,bb[3:]+padding)
            contact=patch^raw
            if D.volume(contact)<=1e-12:raise ValueError('No finite fixed cavity contact')
            floor=float(contact.bounding_box()[2])+.00001
            patch=patch.trim_by_plane((0,0,1),floor)
            if not (void-patch).is_empty():raise ValueError('Seating trim fails to fill the complete void')
            detail=dict(positive_void_file=case['positive_void_file'],positive_void_volume_mm3=D.volume(void),
                        void_coverage_missing_volume_mm3=D.volume(void-patch),padding_mm=padding,
                        source_diagnosis='Measured enclosed void adjacent to an existing permanent fixed micro-reinforcement; explicitly filled, never silently discarded.',
                        original_void_inside_fraction=case['original']['void_inside_fraction'],
                        raw_union_void_inside_fraction=case['fresh_raw']['void_inside_fraction'])
        bb=np.asarray(patch.bounding_box());probe=bb[2]+.00005
        first=patch.trim_by_plane((0,0,-1),-probe);first_contact=first^occupied_reference
        if D.volume(first_contact)<=1e-12 or abs(first_contact.bounding_box()[2]-bb[2])>1e-8:
            raise ValueError('Repair begins without finite earlier fixed-material contact: '+case['id'])
        prefix_pieces=[p for p in first.decompose()if D.volume(p)>1e-15]
        if len(prefix_pieces)!=1:raise ValueError('Repair first prefix is disconnected')
        after=canonical_probe(raw+patch)
        if not after['passed']:raise ValueError('Local strict canonical repair failed: '+case['id']+' '+str(after))
        path=OUT/f'{case["id"]}_fixed_fill.npz';D.save(path,patch);patches.append(patch)
        matched=case['nearby_added_surface_matches']
        owners=[dict(owner=x['owner'],index=x['index'],kind=x['kind'],temporary=x['temporary'])for x in matched]
        rows.append(dict(id=case['id'],kind='fixed_micro_reinforcement',temporary=False,
            repair_purpose=case['kind'],mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),
            target_mm=((bb[:3]+bb[3:])/2).tolist(),root_mm=[float((bb[0]+bb[3])/2),float((bb[1]+bb[4])/2),float(bb[2])],
            bounds_mm=bb.tolist(),volume_mm3=D.volume(patch),
            signed_raw_difference_volume_mm3=float((patch-raw).volume()),
            actual_added_volume_over_occupied_context_mm3=D.volume(patch-occupied_reference),
            occupied_context_definition=('Largest positive context component; complete smaller positive lobe proved contained in it' if case['kind']=='original_rim_edge' else 'Complete raw fixed context with its actual signed cavity boundary'),
            signed_original_context_intersection_volume_mm3=float((patch^original).volume()),
            signed_raw_context_intersection_volume_mm3=float((patch^raw).volume()),
            occupied_context_contact_volume_mm3=D.volume(patch^occupied_reference),
            first_prefix_height_mm=probe,first_prefix_contact_volume_mm3=D.volume(first_contact),
            first_prefix_contact_fraction=D.volume(first_contact)/D.volume(first),first_prefix_components=1,
            nearby_existing_fixed_context=owners,neck_count=0,
            local_before_strict_probe=before,local_after_strict_probe=after,**detail))
        print(case['id'],'local strict pass','added occupied-set volume',rows[-1]['actual_added_volume_over_occupied_context_mm3'],flush=True)
    aggregate=mf.Manifold.batch_boolean(patches,mf.OpType.Add);target=OUT/'fixed_precision_repairs.npz';D.save(target,aggregate)
    if any(D.sha(ROOT/p)!=h for p,h in input_hashes.items()):raise ValueError('Scoped input changed')
    report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,
        diagnostic_only=True,source_models_modified=False,repair_count=len(rows),units='mm',
        aggregate_mesh_file=str(target.relative_to(ROOT)),aggregate_sha256=D.sha(target),
        aggregate_volume_mm3=D.volume(aggregate),repairs=rows,
        input_sha256=input_hashes,source_sha256=D.sha(__file__),elapsed_seconds=time.monotonic()-start)
    (ROOT/'reports/fixed_precision_repair_candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Candidate set ready',len(rows),D.volume(aggregate),flush=True)

if __name__=='__main__':main()
