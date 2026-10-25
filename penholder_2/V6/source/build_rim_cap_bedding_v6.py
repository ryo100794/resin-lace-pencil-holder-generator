#!/usr/bin/env python3
"""Declared positive fixed cap pads for all measured final upper-rim starts."""
from pathlib import Path
import json, time
import numpy as np
import manifold3d as mf
import trimesh
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe

ROOT=D.ROOT
OUT=ROOT/'cache/rim_cap_bedding_v6'
REPORT=ROOT/'reports/rim_cap_bedding_candidates_v6.json'
EXPECTED='6d41bedfff68de23a6b39cf6a1abdf89c4aa9c27e848ca6ad6be7bd2d1b80e8c'

def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def positive(m):
    if m.is_empty():return 0.
    v=float(m.volume())
    if v < -1e-13:raise ValueError('Unexpected signed negative solid')
    return max(0.,v)

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'cache/print_ready_v6.npz',ROOT/'reports/custom_supports_validation.json',ROOT/'reports/final_rim_start_attribution.json',ROOT/'cache/custom_holder_supports/holder_supports.json',Path(__file__)]
    inputs={str(p.relative_to(ROOT)):D.sha(p)for p in paths}
    if inputs['cache/print_ready_v6.npz']!=EXPECTED:raise ValueError('Unexpected final diagnostic model')
    validation=json.loads(paths[1].read_text());events=[r for r in validation['rows']if r['material_birth']and not r['plate_contact']]
    attribution=json.loads(paths[2].read_text())['rows'];holder=json.loads(paths[3].read_text());parts={r['index']:r for r in holder['parts']}
    if len(events)!=51:raise ValueError('Unexpected final start inventory')
    fulls={};rems={};groups={};footprints=[]
    print('Loading final strict mesh once; extracting one upper rim band',flush=True)
    model=D.load(paths[0]);band=model.trim_by_plane((0,0,1),99.35).trim_by_plane((0,0,-1),-100.15)
    print('Upper band triangles',band.num_tri(),flush=True)
    del model
    for k,(event,old)in enumerate(zip(events,attribution)):
        if event['vertex_index']!=old['vertex_index']:raise ValueError('Attribution order mismatch')
        matches=old['matching_added_support_surfaces']
        if len(matches)!=1:raise ValueError('Provisional owner ambiguous')
        index=matches[0]['support_index'];row=parts[index]
        if index not in fulls:
            for key,cache in [('full_mesh_file',fulls),('removable_mesh_file',rems)]:
                path=paths[3].parent/row[key];inputs[str(path.relative_to(ROOT))]=D.sha(path);cache[index]=D.load(path)
        full=fulls[index];p=np.asarray(event['position_mm']);mm=full.to_mesh64();tri=np.asarray(mm.vert_properties)[:,:3][np.asarray(mm.tri_verts)]
        nearest=trimesh.triangles.closest_point(tri,np.repeat(p[None,:],len(tri),axis=0));distance=float(np.linalg.norm(nearest-p,axis=1).min())
        if distance>1e-4:raise ValueError(f'Owner {index} does not contain event {k}: {distance}')
        join=event['surface_join_z_mm'];birth=event['birth_z_mm'];probe=join-1e-7
        lo=p-[.18,.18,.08];hi=p+[.18,.18,0.];hi[2]=join+.08
        context=band^D.box(lo,hi)
        prefix=context.trim_by_plane((0,0,-1),-probe)
        choices=[]
        for component in prefix.decompose():
            vol=float(component.volume())
            if vol<=1e-14:continue
            v=np.asarray(component.to_mesh64().vert_properties)[:,:3]
            dist=float(np.linalg.norm(v-p,axis=1).min())
            choices.append((dist,component,vol))
        choices.sort(key=lambda x:x[0])
        if not choices or choices[0][0]>1e-6:raise ValueError(f'No actual prefix lobe for event{k}')
        dist,lobe,vol=choices[0];bb=np.asarray(lobe.bounding_box())
        if np.any(bb[:2]<=lo[:2]+1e-7)or np.any(bb[3:5]>=hi[:2]-1e-7)or bb[2]<=lo[2]+1e-7:raise ValueError(f'Incomplete lobe crop{k}')
        footprint=dict(event_index=k,support_index=index,vertex_index=event['vertex_index'],position_mm=p.tolist(),birth_z_mm=birth,join_z_mm=join,prejoin_probe_z_mm=probe,prejoin_lobe_volume_mm3=vol,bounds_mm=bb.tolist(),actual_owner_surface_distance_mm=distance,crop_contains_complete_lobe=True)
        footprints.append(footprint);groups.setdefault(index,[]).append(footprint)
        print('Footprint',k,index,'dimensions',list(bb[3:]-bb[:3]),flush=True)
    dump(OUT/'footprints.json',dict(source_mesh_sha256=EXPECTED,rows=footprints))
    repairs=[];patches=[]
    for index,rows in sorted(groups.items()):
        row=parts[index];full=fulls[index];rem=rems[index]
        local_trials=[]
        for padding in [.08,.12,.16]:
            bb=np.asarray([r['bounds_mm']for r in rows]);lo=bb[:,:3].min(axis=0)-[padding,padding,.03];hi=bb[:,3:].max(axis=0)+[padding,padding,padding]
            # Select a seating plane below every still-live nonplate event. This
            # uses the complete authoritative inventory, not a sphere heuristic.
            while True:
                active=[r for r in events if r['birth_z_mm']<=lo[2]+.00005<r['surface_join_z_mm']]
                if not active:break
                lo[2]=min(r['birth_z_mm']for r in active)-.01
            lo=lo.astype(np.float32).astype(float);hi=hi.astype(np.float32).astype(float)
            patch=D.box(lo,hi);floor=float(lo[2]);probe=floor+.00005
            prefix=patch.trim_by_plane((0,0,-1),-probe)
            cap=full-rem;contact=prefix^cap
            if positive(contact)<=1e-12 or abs(contact.bounding_box()[2]-floor)>1e-8:raise ValueError(f'H{index} pad has no retained first-face seat')
            source_prefix=full.trim_by_plane((0,0,-1),-probe);source_comps=[s for s in source_prefix.decompose()if float(s.volume())>1e-12]
            if len(source_comps)!=1:raise ValueError(f'H{index} existing support prefix disconnected')
            if not(patch^rem).is_empty():raise ValueError(f'H{index} pad enters removable')
            gap=float(patch.min_gap(rem,1.))
            axis=np.asarray(row['neck_axes'][1]);target=np.asarray(row['target_mm']);pv=np.asarray(patch.to_mesh64().vert_properties)[:,:3];offset=float(((pv-target)@axis).max())
            if offset>=.075:raise ValueError(f'H{index} pad leaves permanent contact head: {offset}')
            # Rebuild only a bounded actual final neighborhood for local strict gate.
            clo=lo-[.2,.2,.1];chi=hi+[.2,.2,.1];context=band^D.box(clo,chi)
            context_path=OUT/f'H{index:04d}_final_context.npz';D.save(context_path,context)
            before=canonical_probe(context.as_original());after=canonical_probe((context+patch).as_original())
            local_trials.append(dict(side_and_top_padding_mm=padding,before=before,after=after))
            if after['passed']:break
        else:raise ValueError(f'H{index} bounded pad trials failed: {local_trials}')
        ident=f'cap_bedding_H{index:04d}';path=OUT/f'{ident}.npz';D.save(path,patch);patches.append(patch)
        # At the chosen earlier probe the complete source audit has no live
        # positive nonplate component anywhere. Finite source contact is rooted.
        actual_contact=prefix^context
        if positive(actual_contact)<=1e-12 or abs(actual_contact.bounding_box()[2]-floor)>1e-8:raise ValueError('No actual final earlier seating')
        record=dict(id=ident,kind='fixed_micro_reinforcement',temporary=False,
            repair_purpose='uniform_fixed_rim_contact_cap_bedding',affected_support_index=index,
            startup_dependency_support_index=index,mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),
            target_mm=((lo+hi)/2).tolist(),root_mm=[float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),floor],
            bounds_mm=list(patch.bounding_box()),volume_mm3=float(patch.volume()),actual_added_volume_over_occupied_context_mm3=positive(patch-context),
            maximum_box_dimension_mm=float((hi-lo).max()),neck_count=0,
            footprint_side_padding_mm=padding,earlier_seating_depth_mm=.03,upper_padding_mm=padding,local_dimension_trials=local_trials,
            affected_actual_final_events=rows,full_prejoin_footprints_enclosed=True,
            source_support_removable_gap_mm=gap,maximum_outward_axis_offset_from_contact_mm=offset,
            earlier_support_positive_component_count=len(source_comps),
            authoritative_source_audit_has_no_live_nonplate_birth_at_first_prefix=True,
            first_prefix_height_mm=probe,first_prefix_contact_volume_mm3=positive(contact),
            first_prefix_contact_fraction=positive(contact)/positive(prefix),actual_final_first_prefix_contact_volume_mm3=positive(actual_contact),
            final_context_file=str(context_path.relative_to(ROOT)),final_context_sha256=D.sha(context_path),
            nearby_existing_fixed_context=[dict(owner='holder',index=index)],
            local_before_strict_probe=before,local_after_strict_probe=after,
            existing_temporary_and_cut_geometry_unchanged=True,
            persistent_fixed_contact=dict(holder_support_indices=[index],nominal_retained_stock_used=True,contact_volume_mm3=positive(contact),first_prefix_height_mm=probe,first_contact_bottom_mm=float(contact.bounding_box()[2]),patch_bottom_mm=floor,patch_is_convex=True,all_positive_height_prefixes_connected=True,passed=True),
            local_geometry_passed=True)
        repairs.append(record);dump(OUT/'generation_checkpoint.json',dict(repairs=repairs,footprints=footprints))
        print('Pad',index,'strict pass, added',record['actual_added_volume_over_occupied_context_mm3'],'removable gap',gap,flush=True)
    aggregate=mf.Manifold.batch_boolean(patches,mf.OpType.Add);path=OUT/'rim_cap_bedding.npz';D.save(path,aggregate)
    for rel,digest in inputs.items():
        if D.sha(ROOT/rel)!=digest:raise ValueError('Input mutated: '+rel)
    report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,
        source_models_modified=False,repair_count=len(repairs),covered_final_birth_count=len(events),units='mm',
        append_only_fixed_geometry=True,original_model_and_all_temporary_geometry_unchanged=True,
        aggregate_mesh_file=str(path.relative_to(ROOT)),aggregate_sha256=D.sha(path),aggregate_volume_mm3=float(aggregate.volume()),
        total_net_added_material_mm3=sum(r['actual_added_volume_over_occupied_context_mm3']for r in repairs),
        repairs=repairs,input_sha256=inputs,source_sha256=D.sha(__file__),elapsed_seconds=time.monotonic()-started)
    dump(REPORT,report);print('DONE',len(repairs),'pads',len(events),'starts',report['total_net_added_material_mm3'],'mm3',report['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()
