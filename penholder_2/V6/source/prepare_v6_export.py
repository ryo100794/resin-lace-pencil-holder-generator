#!/usr/bin/env python3
"""Bounded representation preparation followed by strict float32 validation.

This does not design, fill, or discard a component. The only approximation
is the declared Manifold 3.5.2 simplify(1e-5 mm) pass count, with summed
surface-displacement bounds, followed by one measured float32 rounding. Original V5 source files
remain unchanged. A separate final gate binds the motion transfer to the
actual exported mesh and the complete build-order audit.
"""
from pathlib import Path
import hashlib, importlib.metadata, json
import numpy as np
from scipy.spatial import cKDTree
from build_test_coupon import canonical_float32_mesh

ROOT=Path(__file__).resolve().parent.parent
SIMPLIFY_TOLERANCE_MM=1e-5
SIMPLIFY_PASSES=1
V5_INPUTS={
    'cache/original_v5.npz':'de722a37883d54ad0d9cb8383aff327d08f2885eceab08ae9d96981c37ece348',
    'source/build_holder.py':'e4c3d0cc5fb25dd46df51ac3b710333002e488ce30eee010f29a8b912a56df62',
    'source/prepare_print.py':'d9eea1fa365bc9c022d4d3fa7e67a234ab6f9f73f7e744e1e025ee7e207b4f5e',
    'source/embroidery_technique.py':'64b8f321aea65d72aef75fbb282f2db6a5b89b5ae00eed079614cc257781859b',
    'reports/precision_repairs_scope.json':'515b1687ea98b3fe82b0f4bcb84a276f0c72f920a10029b2cb65f7ef0e899154',
    'reports/print_preparation.json':'309ec4d440d3254cec35d67bd20c30bf8046f7bb6deb3a1badc810ab98898a84',
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def v5_rounding_chain(root=ROOT):
    for name,digest in V5_INPUTS.items():
        if sha(root/name)!=digest:raise ValueError('V5 precision-chain input changed: '+name)
    scope=json.loads((root/'reports/precision_repairs_scope.json').read_text())
    prep=json.loads((root/'reports/print_preparation.json').read_text())
    export=json.loads((root/'reports/export_validation.json').read_text())
    if not scope.get('passed') or scope['preparation_report_sha256']!=V5_INPUTS['reports/print_preparation.json']:
        raise ValueError('Original fixed-repair scope evidence is not valid')
    if export['source_sha256']!=V5_INPUTS['cache/original_v5.npz']:
        raise ValueError('Original exported input differs from the V5 lineage')
    bounds=np.asarray(prep['bounds_mm'],float)
    maximum=np.max(np.abs(bounds),axis=0)
    # All original source coordinates remain in the same binary exponent bins
    # even after allowing a full float32 ULP around the released bounds.
    enlarged=maximum+np.abs(np.spacing(maximum.astype(np.float32)).astype(float))
    half_ulp=.5*np.abs(np.spacing(enlarged.astype(np.float32)).astype(float))
    error=float(np.linalg.norm(half_ulp))
    return dict(passed=True,prior_moving_surface_quantization_count=1,
        prior_per_surface_rounding_upper_bound_mm=error,
        coordinate_half_ulp_upper_bounds_mm=half_ulp.tolist(),
        source_lineage_sha256=V5_INPUTS,
        export_validation_sha256=sha(root/'reports/export_validation.json'),
        minimum_fixed_repair_distance_to_original_moving_envelopes_mm=scope['minimum_distance_to_moving_envelopes_mm'],
        rationale='V5 sphere caches enter build_holder as Mesh64; wall-only simplification precedes their assembly. E.merge is a float64 Boolean. Diagnostic float32 meshes in prepare_print never replace the float64 solid. All added V5 repair primitives/fills are separately scoped outside moving/bearing/neck geometry. Only the final V5 mesh is rounded and saved.')

def exact_oriented_surface_equal(v0,f0,v1,f1):
    """Complete exact oriented triangle multiset, with multiplicity retained."""
    if np.array_equal(v0,v1) and np.array_equal(f0,f1):
        return dict(passed=True,method='Exact identical vertex and face arrays')
    if len(f0)!=len(f1):return dict(passed=False,reason='Triangle count changed')
    unique,inverse=np.unique(v0,axis=0,return_inverse=True)
    tree=cKDTree(unique);distance,index=tree.query(v1,workers=1)
    if np.any(distance):return dict(passed=False,reason='Vertex coordinates changed')
    def orient(f):
        rotation=np.argmin(f,axis=1)
        return np.take_along_axis(f,(rotation[:,None]+np.arange(3)[None,:])%3,axis=1)
    t0=orient(inverse[f0]);t1=orient(index[f1])
    o0=np.lexsort((t0[:,2],t0[:,1],t0[:,0]));o1=np.lexsort((t1[:,2],t1[:,1],t1[:,0]))
    return dict(passed=bool(np.array_equal(t0[o0],t1[o1])),
        method='Exact complete oriented triangle coordinate multiset, preserving multiplicity and winding')

def prepare(solid,raw_path,root=ROOT):
    version=importlib.metadata.version('manifold3d')
    if version!='3.5.2':raise ValueError('This preparation is pinned to manifold3d 3.5.2')
    chain=v5_rounding_chain(root)
    before=solid.to_mesh64()
    source_vertices=np.asarray(before.vert_properties[:,:3],float)
    with np.load(raw_path,allow_pickle=False) as saved:
        if not (np.array_equal(saved['vertices'],source_vertices)
                and np.array_equal(saved['faces'],before.tri_verts)):
            raise ValueError('Passed solid does not exactly match the saved raw union')
    simplified=solid
    provenance_passes=[];pass_bounds=[]
    for index in range(SIMPLIFY_PASSES):
        pass_input=simplified.to_mesh64()
        reset=simplified.as_original();reset_mesh=reset.to_mesh64()
        proof=exact_oriented_surface_equal(
            np.asarray(pass_input.vert_properties[:,:3]),pass_input.tri_verts,
            np.asarray(reset_mesh.vert_properties[:,:3]),reset_mesh.tri_verts)
        if not proof['passed']:
            raise ValueError('Resetting face provenance changed the complete source geometry in pass '+str(index+1))
        provenance_passes.append(dict(pass_index=index+1,**proof))
        pass_bounds.append(max(SIMPLIFY_TOLERANCE_MM,float(reset.get_tolerance())))
        simplified=reset.simplify(SIMPLIFY_TOLERANCE_MM)
        del pass_input,reset_mesh,reset
    effective=sum(pass_bounds)
    provenance=dict(passed=True,passes=provenance_passes,
        method='Each reset preserves the complete exact oriented triangle coordinate multiset, including multiplicity and winding.')
    mm=simplified.to_mesh64()
    new_vertices=np.asarray(mm.vert_properties[:,:3],float)
    tree=cKDTree(source_vertices)
    maximum_subset_distance=0.
    for start in range(0,len(new_vertices),65536):
        distances,_=tree.query(new_vertices[start:start+65536],workers=1)
        maximum_subset_distance=max(maximum_subset_distance,float(distances.max(initial=0.)))
    if maximum_subset_distance!=0.:raise ValueError('Simplify returned a vertex outside the original vertex set')
    del tree,source_vertices,before
    simplified_path=root/'cache/assembled_v6_prepared_float64.npz'
    np.savez_compressed(simplified_path,vertices=new_vertices,faces=mm.tri_verts)
    vertices,faces,topology=canonical_float32_mesh(simplified)
    final_round=float(topology['maximum_vertex_rounding_error_mm'])
    total=chain['prior_per_surface_rounding_upper_bound_mm']+effective+final_round
    report=dict(passed=True,manifold_version=version,source_union_sha256=sha(raw_path),
        simplified_float64_sha256=sha(simplified_path),
        source_chain=chain,simplification_requested_tolerance_mm=SIMPLIFY_TOLERANCE_MM,
        simplification_pass_count=SIMPLIFY_PASSES,
        individual_simplification_surface_bounds_mm=pass_bounds,
        source_manifold_tolerance_mm=float(solid.get_tolerance()),
        effective_simplification_surface_displacement_bound_mm=effective,
        output_vertices_exact_subset_of_input_vertices=True,
        provenance_reset_preserves_complete_oriented_surface=provenance,
        maximum_retained_vertex_distance_mm=maximum_subset_distance,
        final_single_float32_rounding_error_mm=final_round,
        complete_source_to_export_surface_error_upper_bound_mm=total,
        complete_two_surface_error_upper_bound_mm=2*total,
        source_solid_exactly_matches_saved_raw_union=True,
        explicit_component_deletion_step=False,explicit_cavity_filling_step=False,
        operation_order=['Fresh union of unchanged V5 and declared additions in float64',
            'For each in-memory pass: reset face provenance and prove exact complete oriented triangle multiset equality',
            str(SIMPLIFY_PASSES)+' bounded simplify operation(s); sum their surface-displacement bounds','One float32 rounding',
            'Exact coordinate welding and collapsed-index-face removal only',
            'Strict complete-surface topology validation'],
        bound_source='manifold3d 3.5.2 Manifold.simplify docstring: output vertices are a subset of the input; all surfaces move less than the effective tolerance.',
        documentation_url='https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html',
        unit_note='1e-5 mm is 10 nanometres, or 0.01 micrometres.',
        preparation_source_sha256=sha(__file__),
        canonical_validator_sha256=sha(root/'source/build_test_coupon.py'))
    return vertices,faces,topology,report
