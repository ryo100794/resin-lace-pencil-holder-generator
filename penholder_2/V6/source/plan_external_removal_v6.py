#!/usr/bin/env python3
"""Plan actual V6 external support removal in the complete holder.

Every chosen move is checked while all not-yet-removed supports, all nominal
cut stumps, permanent micro-braces, and the original V5 model are present.
The convex pieces are conservative continuous translation bounds. Body cuts
are explicit 0.05 mm kerfs, never an implicit disappearance of a branch.
This is a geometric removal plan, not a physical cutter/hand-access test.
"""
from pathlib import Path
import argparse, hashlib, io, json, math, time
import numpy as np
import manifold3d as mf
from rtree import index as rtree_index
import custom_holder_supports as H
from build_holder import SPHERES, sphere_transform

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/external_removal_v6'
TOL=1e-8
INPUT_HASHES={}
SCENE_MIN_Z=None
SCENE_MAX_Z=None
ROUTE_SEEDS={}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frozen_bytes(path):
    path=Path(path);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    key=str(path.relative_to(ROOT))
    if key in INPUT_HASHES and INPUT_HASHES[key]!=digest:
        raise ValueError('Input changed between loads: '+key)
    INPUT_HASHES[key]=digest
    return data

def metadata(path):return json.loads(frozen_bytes(path))

def transform_point(p,t):
    return (t[:,:3]@np.asarray(p,float)+t[:,3]).tolist()

def transform_direction(d,t):
    return (t[:,:3]@np.asarray(d,float)).tolist()

def solid(path,t=None):
    with np.load(io.BytesIO(frozen_bytes(path)),allow_pickle=False) as d:
        vertices=np.asarray(d['vertices'],float);faces=np.asarray(d['faces'],np.uint32)
    m=mf.Manifold(mf.Mesh64(vertices,faces))
    if m.status()!=mf.Error.NoError:raise ValueError('Invalid actual input solid: '+str(path))
    return m.transform(t) if t is not None else m

def load_parts():
    records=[];permanent=[];inputs=[];all_full=[]
    partition_path=ROOT/'reports/sphere_removal_partition_v6.json'
    partition=metadata(partition_path)
    if partition.get('passed') is not True:
        raise ValueError('Finite sphere endpoint-cut partition has not passed')
    partition_source=ROOT/'source/sphere_removal_partition_v6.py'
    frozen_bytes(partition_source)
    if INPUT_HASHES[str(partition_source.relative_to(ROOT))]!=partition['source_sha256']:
        raise ValueError('Endpoint-cut partition implementation changed')
    for path,digest in partition['inputs'].items():
        frozen_bytes(ROOT/path)
        if INPUT_HASHES[path]!=digest:raise ValueError('Endpoint-cut partition source changed: '+path)
    partition_rows={(r['layer_count'],r['support_index'],r['placement']):r
        for r in partition['records'] if r['placement'] is not None}
    def derived(entry):
        mesh=solid(ROOT/entry['path'])
        if INPUT_HASHES[entry['path']]!=entry['sha256']:
            raise ValueError('Derived finite-cut mesh changed: '+entry['path'])
        return mesh
    hp=ROOT/'cache/custom_holder_supports/holder_supports.json'
    if not hp.exists():hp=ROOT/'reports/custom_holder_supports.json'
    h=metadata(hp)
    if h.get('generation_checks_passed',h.get('passed')) is not True:
        raise ValueError('Holder generation has not passed')
    if h.get('unresolved_selected_candidates') or h.get('foundation_failures'):
        raise ValueError('Holder generation contains unresolved design candidates')
    inputs.append(hp)
    for row in h['parts']:
        folder=ROOT/'cache/custom_holder_supports'
        full=solid(folder/row['full_mesh_file']);inputs.append(folder/row['full_mesh_file'])
        all_full.append(full)
        if row['kind']=='fixed_micro_reinforcement':permanent.append(full);continue
        rem=solid(folder/row['removable_mesh_file']);inputs.append(folder/row['removable_mesh_file'])
        permanent.append(full-rem)
        if rem.is_empty() or H.volume(rem)<1e-12:continue
        pieces=[solid(folder/x) for x in row.get('generated_convex_sweep_meshes',[])]
        if not pieces:pieces=[rem]
        records.append(dict(id=f'H{row["index"]:04d}',group='holder',source_index=row['index'],
            kind=row['kind'],full=full,removable=rem,pieces=pieces,record=row,
            root=np.asarray(row['root_mm']),target=np.asarray(row['target_mm'])))
    for placement,s in enumerate(SPHERES):
        n=s['layers'];folder=ROOT/'cache/custom_sphere_supports'
        rp=folder/f'sphere_{n}layers_supports.json';report=metadata(rp)
        if report.get('passed') is not True or report.get('finalization_complete') is not True:
            raise ValueError(f'Sphere {n} generation has not passed')
        if rp not in inputs:inputs.append(rp)
        t=sphere_transform(s)
        for row in report['supports']:
            full=solid(folder/row['mesh_file'],t)
            all_full.append(full)
            if folder/row['mesh_file'] not in inputs:inputs.append(folder/row['mesh_file'])
            if row['kind']=='fixed_micro_reinforcement':permanent.append(full);continue
            cut_record=partition_rows[(n,row['support_index'],placement)]
            if cut_record.get('passed') is not True:raise ValueError('Failed placed endpoint-cut partition')
            for field,key in [('source_full_mesh_file','source_full_sha256'),('source_maximum_mesh_file','source_maximum_sha256')]:
                frozen_bytes(ROOT/cut_record[field])
                if INPUT_HASHES[cut_record[field]]!=cut_record[key]:
                    raise ValueError('Endpoint partition source stock changed: '+cut_record[field])
            if cut_record['source_full_mesh_file']!=str((folder/row['mesh_file']).relative_to(ROOT)):
                raise ValueError('Endpoint partition is assigned to the wrong printed part')
            paths=cut_record['paths']
            rem=derived(paths['removable']);permanent.append(derived(paths['retained']))
            kerf=derived(paths['kerf']);cutters=[derived(x) for x in paths['cutters']]
            fragments=[derived(x) for x in paths['fragments']]
            pieces=[derived(x) for x in paths['proxies']]
            if not pieces and not rem.is_empty():pieces=[rem]
            records.append(dict(id=f'S{n}_A{int(s["theta_deg"]):03d}_{row["support_index"]:03d}',
                group='sphere',placement=placement,layers=n,azimuth_deg=s['theta_deg'],
                source_index=row['support_index'],kind=row['kind'],full=full,removable=rem,
                pieces=pieces,record=row,transform=t,
                endpoint_kerf=kerf,endpoint_cutters=cutters,endpoint_partition=cut_record,
                precut_fragments=fragments,
                root=np.asarray(transform_point(row['anchor_point'],t)),
                target=np.asarray(transform_point(row['contact_point'],t))))
    return records,permanent,inputs,all_full

class ObstacleField:
    """Static spatial index; removed bodies are disabled, stumps remain."""
    def __init__(self,items,permanent):
        prop=rtree_index.Property();prop.dimension=3
        self.tree=rtree_index.Index(properties=prop)
        self.solids=[];self.owner=[];self.active={x['id'] for x in items}
        # Keep each signed retained solid intact.  Decomposing a solid into
        # separate oriented boundary shells would turn cavities into spurious
        # independent obstacles and lose the finite-cut partition semantics.
        for p in permanent:self.add(p,None)
        for item in items:self.add(item['full'],item['id'])

    def add(self,solid,owner):
        if solid.is_empty():return
        i=len(self.solids);self.solids.append(solid);self.owner.append(owner)
        self.tree.insert(i,tuple(solid.bounding_box()))

    def context(self,excluded):return ObstacleContext(self,excluded,[])

class ObstacleContext:
    def __init__(self,field,excluded,extra):
        self.field=field;self.excluded=excluded;self.extra=extra

    def with_extra(self,extra):return ObstacleContext(self.field,self.excluded,self.extra+extra)

    def obstruction(self,query,original,convex=True):
        if query.is_empty():return 0.
        # Small nearby stumps and still-attached branches cheaply reject
        # blocked departures before the same whole-model intersection test.
        # The obstacle set and positive-volume tolerance are unchanged.
        value=0.
        for i in self.field.tree.intersection(tuple(query.bounding_box())):
            owner=self.field.owner[i]
            if owner is not None and (owner==self.excluded or owner not in self.field.active):continue
            value+=H.volume(query^self.field.solids[i])
            if value>TOL:return value
        for solid in self.extra:
            if H.bbox_intersects(query,solid):
                value+=H.volume(query^solid)
                if value>TOL:return value
        if not (convex and H.ACTIVE_SURFACE is not None and original is H.ACTIVE_SURFACE.solid and not H.ACTIVE_SURFACE.may_hit_convex(query)):
            value+=H.volume(query^original)
        return value

def safe_sweep(pieces,direction,distance,original,obstacles):
    worst=0.
    for piece in pieces:
        bound=H.sweep(piece,direction,distance)
        collision=obstacles.obstruction(bound,original)
        worst=max(worst,collision)
        if collision>TOL:return False,worst
    return True,worst

def candidate_directions(item,center,preferred=()):
    if item['kind']=='plate_perimeter_tack':return [np.array([0.,0.,-1.])]
    out=[np.asarray(d,float) for d in preferred]
    for frag in item['record'].get('removal_fragments',[]):
        direction=np.asarray(frag['direction'],float)
        if item['group']=='sphere':direction=item['transform'][:,:3]@direction
        out.append(direction)
    if item['group']=='sphere':
        t=item['transform']
        local_center=t[:,:3].T@(np.asarray(center)-t[:,3])
        # Prefer the exposed side of the actual hemisphere. Both signs are
        # still ordinary candidates and receive the identical scene checks.
        out.append(t[:,:3]@np.array([0.,0.,1. if local_center[2]>=0 else -1.]))
        radial=np.asarray(center)-t[:,3]
        if np.linalg.norm(radial)>1e-8:out.append(radial/np.linalg.norm(radial))
        out.extend([t[:,:3]@np.array(d,float) for d in
            [(0,0,1),(0,.5,1),(0,-.5,1),(.5,0,1),(-.5,0,1)]])
    out+=H.paths_at(center)
    result=[]
    for d in out:
        d=d/np.linalg.norm(d)
        if not any(np.linalg.norm(d-p)<1e-5 for p in result):result.append(d)
    return result

def route_fragment(item,fragment,pieces,original,obstacles,preferred=(),broad_search=True):
    bb=np.asarray(fragment.bounding_box());center=(bb[:3]+bb[3:])/2
    directions=candidate_directions(item,center,preferred)
    if not broad_search:directions=directions[:4]
    for d in directions:
        distances=[max(.5,float(bb[5]-SCENE_MIN_Z+.05))] if item['kind']=='plate_perimeter_tack' else [20.,35.,55.,90.]
        for travel in distances:
            ok,overlap=safe_sweep(pieces,d,travel,original,obstacles)
            # A longer collinear sweep contains the shorter failed sweep.
            if not ok:break
            legs=[dict(direction=d.tolist(),distance_mm=travel,maximum_swept_intersection_mm3=overlap)]
            if item['kind']!='plate_perimeter_tack':
                # A radial move into the pen cavity is followed by an actual
                # certified upward move out of the open top, not assumed free.
                moved=[p.translate(d*travel) for p in pieces]
                minz=min(p.bounding_box()[2] for p in moved)
                exit_distance=max(0.,SCENE_MAX_Z+1.-minz)
                if exit_distance:
                    ok,overlap=safe_sweep(moved,np.array([0.,0.,1.]),exit_distance,original,obstacles)
                    if not ok:continue
                    legs.append(dict(direction=[0.,0.,1.],distance_mm=exit_distance,
                        maximum_swept_intersection_mm3=overlap))
            end=fragment
            for leg in legs:end=end.translate(np.asarray(leg['direction'])*leg['distance_mm'])
            ebb=end.bounding_box()
            escaped=ebb[5]<SCENE_MIN_Z if item['kind']=='plate_perimeter_tack' else ebb[2]>SCENE_MAX_Z
            if not escaped:continue
            return legs
    return None

def body_partition(rem,cuts):
    """Clip actual stock into interleaved fragments and finite kerf bands."""
    if not cuts:return None
    axis=np.asarray(cuts[0]['plane_normal'],float);length=float(np.linalg.norm(axis))
    if abs(length-1.)>1e-8:return None
    levels=[];widths=[]
    for cut in cuts:
        if np.linalg.norm(np.asarray(cut['plane_normal'])-axis)>1e-8:return None
        width=float(cut['modeled_removed_kerf_width_mm'])
        if not 0.<width<=.05+1e-12:return None
        levels.append(float(cut['plane_offset_mm']));widths.append(width)
    if any(levels[i]+widths[i]/2>=levels[i+1]-widths[i+1]/2 for i in range(len(levels)-1)):return None
    fragments=[];bands=[]
    for k in range(len(levels)+1):
        part=rem
        if k:part=part.trim_by_plane(axis,levels[k-1]+widths[k-1]/2)
        if k<len(levels):part=part.trim_by_plane(-axis,-(levels[k]-widths[k]/2))
        if H.volume(part)>1e-12:fragments.append(part)
    for level,width in zip(levels,widths):
        bands.append(rem.trim_by_plane(axis,level-width/2).trim_by_plane(-axis,-(level+width/2)))
    kerf=H.union(bands);cover=H.union(fragments+[kerf])
    checks=dict(uncovered_actual_removable_mm3=H.volume(rem-cover),
        outside_actual_removable_mm3=H.volume(cover-rem),
        signed_partition_volume_error_mm3=abs(sum(p.volume() for p in fragments)+kerf.volume()-rem.volume()))
    if max(checks.values())>TOL:return None
    return fragments,kerf,checks

def verify_saved_legs(item,fragment,pieces,legs,original,obstacles):
    """Treat old routes as candidates and recheck every continuous move."""
    moved=pieces;end=fragment;verified=[]
    for leg in legs:
        direction=np.asarray(leg['direction'],float);distance=float(leg['distance_mm'])
        if not np.all(np.isfinite(direction)) or not math.isfinite(distance) or distance<0 or abs(np.linalg.norm(direction)-1)>1e-8:return None
        ok,overlap=safe_sweep(moved,direction,distance,original,obstacles)
        if not ok:return None
        verified.append(dict(direction=direction.tolist(),distance_mm=distance,maximum_swept_intersection_mm3=overlap))
        moved=[p.translate(direction*distance) for p in moved];end=end.translate(direction*distance)
    bb=end.bounding_box()
    escaped=bb[5]<SCENE_MIN_Z if item['kind']=='plate_perimeter_tack' else bb[2]>SCENE_MAX_Z
    return verified if escaped else None

def try_body_cuts(item,rem,cuts,original,obstacles,saved_routes=None,preferred=None,broad_search=True):
    partition=body_partition(rem,cuts)
    if partition is None:return None
    fragments,kerf,checks=partition
    if not fragments:return None
    overlap=obstacles.obstruction(kerf,original,convex=False)
    if overlap>TOL:return None
    if saved_routes is not None and len(saved_routes)!=len(fragments):return None
    routes=[]
    for k,fragment in enumerate(fragments):
        others=obstacles.with_extra([p for j,p in enumerate(fragments) if j!=k])
        if saved_routes is not None:
            legs=verify_saved_legs(item,fragment,[fragment],saved_routes[k]['translation_legs'],original,others)
        else:
            directions=preferred[k:k+1] if preferred else []
            legs=route_fragment(item,fragment,[fragment],original,others,directions,broad_search)
        if legs is None:return None
        routes.append((fragment,legs))
    return dict(mode='explicit_body_cuts',cuts=cuts,fragments=routes,
        kerf_volume_mm3=H.volume(kerf),kerf_original_overlap_mm3=H.volume(kerf^original),
        kerf_complete_obstacle_state_overlap_mm3=overlap,body_cut_partition_checks=checks,
        actual_body_kerf=kerf)

def make_plan(item,original,obstacles,broad_search=True):
    rem=item['removable'];pieces=item['pieces']
    endpoint_check=None
    if 'endpoint_partition' in item:
        cutter_overlaps=[obstacles.obstruction(cutter,original,convex=False)
            for cutter in item['endpoint_cutters']]
        kerf_overlap=obstacles.obstruction(item['endpoint_kerf'],original,convex=False)
        if max(cutter_overlaps+[kerf_overlap])>TOL:
            return None,dict(reason='Finite endpoint-cut region intersects the original, retained stock or another remaining support',
                cutter_complete_obstacle_state_overlap_mm3=cutter_overlaps,
                actual_kerf_complete_obstacle_state_overlap_mm3=kerf_overlap)
        endpoint_check=dict(passed=True,finite_cutters_checked=True,
            cutter_complete_obstacle_state_overlap_mm3=cutter_overlaps,
            actual_kerf_complete_obstacle_state_overlap_mm3=kerf_overlap,
            actual_endpoint_kerf_volume_mm3=H.volume(item['endpoint_kerf']))
    def accepted(plan):
        plan['endpoint_cut_validation']=endpoint_check
        return plan,None
    if rem.is_empty() or H.volume(rem)<1e-12:
        if endpoint_check is None:return None,dict(reason='Empty removable body has no declared finite endpoint cuts')
        return accepted(dict(mode='endpoint_cuts_only',cuts=[],fragments=[],
            covered_missing_volume_mm3=0.))
    initial=H.volume(rem^original)
    if initial>TOL:return None,dict(reason='Initial removable material intersects original model',volume_mm3=initial)
    # Assert that a purported collection of convex pieces actually covers
    # the complete nominal removable material before using its sweep.
    missing=H.volume(rem-H.union(pieces))
    if missing>TOL:pieces=[rem]
    seed=ROUTE_SEEDS.get(item['id'])
    if seed is not None:
        plan=None
        if seed['mode']=='whole_piece' and len(seed['fragments'])==1:
            legs=verify_saved_legs(item,rem,pieces,seed['fragments'][0]['translation_legs'],original,obstacles)
            if legs is not None:
                plan=dict(mode='whole_piece',cuts=[],fragments=[(rem,legs)],
                    original_proxy_missing_volume_mm3=missing,proxy_fallback_to_actual_body=missing>TOL,
                    covered_missing_volume_mm3=0. if missing>TOL else missing)
        elif seed['mode']=='explicit_body_cuts':
            plan=try_body_cuts(item,rem,seed['additional_body_cuts'],original,obstacles,
                saved_routes=seed['fragments'])
        if plan is not None:
            plan['candidate_route_revalidated']=True
            plan['route_candidate_source']='previous_run_fully_rechecked_against_current_complete_state'
            return accepted(plan)
    declared=item['record']
    if item['group']=='holder' and declared.get('removal_mode')=='explicit_body_cuts':
        plan=try_body_cuts(item,rem,declared['additional_body_cuts'],original,obstacles,
            preferred=[f['direction'] for f in declared.get('removal_fragments',[])],broad_search=broad_search)
        if plan is not None:
            plan['route_candidate_source']='generation_declared_body_cuts_recomputed_from_actual_current_stock'
            return accepted(plan)
    legs=route_fragment(item,rem,pieces,original,obstacles,broad_search=broad_search)
    if legs is not None:return accepted(dict(mode='whole_piece',cuts=[],fragments=[(rem,legs)],
        original_proxy_missing_volume_mm3=missing,proxy_fallback_to_actual_body=missing>TOL,
        covered_missing_volume_mm3=0. if missing>TOL else missing))
    if not broad_search:
        return None,dict(reason='Preferred departures blocked; deferred until easier neighbours are removed')
    axis=item['target']-item['root'];length=float(np.linalg.norm(axis))
    if length<1e-6:return None,dict(reason='No clear whole-piece route')
    axis/=length;mesh=rem.to_mesh64();v=np.asarray(mesh.vert_properties[:,:3]);lo,hi=(v@axis).min(),(v@axis).max()
    for count in sorted(set([2,3,max(3,math.ceil((hi-lo)/1.8)),max(4,math.ceil((hi-lo)/1.2))])):
        levels=np.linspace(lo,hi,count+1)[1:-1];cuts=[]
        for level in levels:
            frac=(level-np.dot(axis,item['root']))/length
            cuts.append(dict(plane_normal=axis.tolist(),plane_offset_mm=float(level),
                centerline_reference_mm=(item['root']+frac*(item['target']-item['root'])).tolist(),
                modeled_removed_kerf_width_mm=.05))
        plan=try_body_cuts(item,rem,cuts,original,obstacles)
        if plan is not None:return accepted(plan)
    return None,dict(reason='No certified whole or segmented route with remaining model/supports present')

def save_plan(item,plan,sequence):
    fragments=[]
    for k,(part,legs) in enumerate(plan['fragments']):
        name=f'{item["id"]}_fragment_{k:02d}.npz';H.save(OUT/name,part)
        end=part
        for leg in legs:end=end.translate(np.asarray(leg['direction'])*leg['distance_mm'])
        fragments.append(dict(mesh_file=str((OUT/name).relative_to(ROOT)),sha256=sha(OUT/name),
            bounds_mm=list(part.bounding_box()),translation_legs=legs,terminal_bounds_mm=list(end.bounding_box()),
            exits_initial_scene_below_or_above=True))
    body_kerf=None
    if 'actual_body_kerf' in plan:
        path=OUT/f'{item["id"]}_body_kerf.npz';H.save(path,plan['actual_body_kerf'])
        body_kerf=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    original=item['record'];t=item.get('transform')
    cuts=original.get('cut_points',original.get('neck_centers_mm',[]))
    if t is not None:cuts=[transform_point(p,t) for p in cuts]
    normals=original.get('cut_plane_normals',original.get('neck_axes',[]))
    if t is not None:normals=[transform_direction(p,t) for p in normals]
    endpoint=item.get('endpoint_partition')
    if endpoint is not None:
        cuts=[cut['center_mm'] for cut in endpoint['cuts']]
        normals=[cut['axis_toward_free_body'] for cut in endpoint['cuts']]
    return dict(id=item['id'],sequence_index=sequence,group=item['group'],
        source_index=item['source_index'],kind=item['kind'],layers=item.get('layers'),
        azimuth_deg=item.get('azimuth_deg'),root_mm=item['root'].tolist(),target_mm=item['target'].tolist(),
        endpoint_cut_centers_mm=cuts,endpoint_cut_plane_normals=normals,
        endpoint_cut_bands=(endpoint['cuts'] if endpoint is not None else []),
        endpoint_cut_partition=(dict(source_report='reports/sphere_removal_partition_v6.json',
            paths=endpoint['paths'],checks=endpoint['checks']) if endpoint is not None else None),
        endpoint_cut_validation=plan.get('endpoint_cut_validation'),
        mode=plan['mode'],additional_body_cuts=plan['cuts'],
        fragments=fragments,kerf_volume_mm3=plan.get('kerf_volume_mm3',0.),
        kerf_original_overlap_mm3=plan.get('kerf_original_overlap_mm3',0.),
        kerf_complete_obstacle_state_overlap_mm3=plan.get('kerf_complete_obstacle_state_overlap_mm3',0.),
        body_cut_partition_checks=plan.get('body_cut_partition_checks'),actual_body_kerf=body_kerf,
        candidate_route_revalidated=plan.get('candidate_route_revalidated',False),
        route_candidate_source=plan.get('route_candidate_source','new_current_state_search'),
        original_proxy_missing_volume_mm3=plan.get('original_proxy_missing_volume_mm3'),
        proxy_fallback_to_actual_body=plan.get('proxy_fallback_to_actual_body',False),
        covered_missing_volume_mm3=plan.get('covered_missing_volume_mm3'),passed=True)

def main():
    global SCENE_MIN_Z,SCENE_MAX_Z,ROUTE_SEEDS
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int)
    ap.add_argument('--assembly-context',type=Path,default=ROOT/'reports/custom_support_assembly.json',
        help='Passed final assembly, or an explicitly scoped frozen nominal-design context. Actual part/aggregate equality is always checked below.')
    ap.add_argument('--reuse-plans',type=Path,
        help='Immutable candidate plan snapshot. Every actual cut, continuous route and scene exit is checked again against the current complete state.')
    a=ap.parse_args()
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    for p in [Path(__file__),ROOT/'source/custom_holder_supports.py',ROOT/'source/build_holder.py',
              ROOT/'source/requirements.txt']:
        frozen_bytes(p)
    if a.reuse_plans:
        seed=metadata(a.reuse_plans)
        ROUTE_SEEDS={row['id']:row for row in seed['plans']}
        if len(ROUTE_SEEDS)!=len(seed['plans']):raise ValueError('Duplicate candidate route IDs')
    items,permanent,inputs,all_full=load_parts()
    assembly=metadata(a.assembly_context)
    if assembly.get('passed') is not True:raise ValueError('No passed assembly to bind removal geometry to')
    for row in assembly['inputs']:
        frozen_bytes(ROOT/row['path'])
        if INPUT_HASHES[row['path']]!=row['sha256']:
            raise ValueError('Declared source context input changed: '+row['path'])
    additions_path=ROOT/'cache/added_supports_v6.npz';retained_path=ROOT/'cache/added_retained_v6.npz'
    additions=solid(additions_path);maximum_retained=solid(retained_path)
    if INPUT_HASHES[str(additions_path.relative_to(ROOT))]!=assembly['added_support_mesh_sha256']:
        raise ValueError('Assembled additions changed after assembly validation')
    if INPUT_HASHES[str(retained_path.relative_to(ROOT))]!=assembly['retained_geometry_sha256']:
        raise ValueError('Maximum retained geometry changed after assembly validation')
    full_union=H.union(all_full);nominal_permanent=H.union(permanent)
    extra_bounds=full_union.bounding_box()
    missing_from_parts=H.volume(additions-full_union);extra_in_parts=H.volume(full_union-additions)
    nominal_outside_maximum=H.volume(nominal_permanent-maximum_retained)
    binding=dict(passed=max(missing_from_parts,extra_in_parts,nominal_outside_maximum)<=1e-7,
        assembled_additions_missing_from_individual_full_parts_mm3=missing_from_parts,
        individual_full_parts_extra_over_assembled_additions_mm3=extra_in_parts,
        nominal_retained_material_outside_maximum_retained_aggregate_mm3=nominal_outside_maximum,
        aggregate_set_difference_tolerance_mm3=1e-7)
    if not binding['passed']:raise ValueError('Removal geometry does not match assembly: '+str(binding))
    print('Individual parts and nominal retained state match assembly aggregates.',flush=True)
    del all_full,full_union,nominal_permanent,additions,maximum_retained
    original_path=ROOT/'cache/original_v5.npz';original_digest=sha(original_path)
    original_entry=next((x for x in assembly['inputs'] if x['path']=='cache/original_v5.npz'),None)
    if original_entry is None or original_entry['sha256']!=original_digest:
        raise ValueError('Original removal obstacle differs from the assembled original model')
    surface=H.Surface(original_path);H.ACTIVE_SURFACE=surface;original=surface.solid
    if sha(original_path)!=original_digest:raise ValueError('Original model changed while loading')
    INPUT_HASHES[str(original_path.relative_to(ROOT))]=original_digest
    original_bounds=original.bounding_box()
    SCENE_MIN_Z=min(float(original_bounds[2]),float(extra_bounds[2]))
    SCENE_MAX_Z=max(float(original_bounds[5]),float(extra_bounds[5]))
    field=ObstacleField(items,permanent)
    # Prefer outer holder branches first, each class in reverse creation order.
    seed_order={key:index for index,key in enumerate(ROUTE_SEEDS)}
    remaining=sorted(items,key=lambda x:(0,seed_order[x['id']],0,x['id']) if x['id'] in seed_order else
        (1,0 if x['group']=='holder' else 1,-x['source_index'],x['id']))
    plans=[];failures=[];initial_count=len(items);broad_search=False
    while remaining:
        progress=False;failures=[]
        for item in list(remaining):
            obstacles=field.context(item['id'])
            plan,error=make_plan(item,original,obstacles,broad_search=broad_search)
            if plan is None:
                failures.append(dict(id=item['id'],**error))
                print('Deferred',item['id'],error['reason'],flush=True)
                continue
            row=save_plan(item,plan,len(plans));plans.append(row)
            remaining=[x for x in remaining if x is not item];progress=True
            field.active.remove(item['id'])
            print('Removal',len(plans),'/',initial_count,item['id'],row['mode'],flush=True)
            checkpoint=dict(complete=False,initial_support_count=initial_count,plans=plans,
                remaining_ids=[x['id'] for x in remaining])
            (OUT/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
            if a.limit and len(plans)>=a.limit:progress=False;break
        if not progress:
            if not broad_search and not (a.limit and len(plans)>=a.limit):
                broad_search=True
                print('Preferred-route phase complete; searching every remaining whole/segmented route.',flush=True)
                continue
            break
    passed=not remaining
    changed=[p for p,digest in INPUT_HASHES.items() if sha(ROOT/p)!=digest]
    if changed:raise ValueError('Certificate input changed during planning: '+str(changed))
    report=dict(passed=passed,units='mm',build_direction=[0,0,1],initial_support_count=initial_count,
        source_context_file=str(a.assembly_context.relative_to(ROOT)),
        source_context_type=assembly.get('context_type','validated_print_assembly'),
        reused_candidate_snapshot=(str(a.reuse_plans.relative_to(ROOT)) if a.reuse_plans else None),
        candidate_routes_fully_revalidated_count=sum(bool(p['candidate_route_revalidated']) for p in plans),
        search_order='Revalidated prior candidates, then short preferred-route passes, then full direction and body-cut search on remaining items; no failed item is silently omitted.',
        assembly_geometry_binding=binding,
        initial_scene_z_bounds_mm=[SCENE_MIN_Z,SCENE_MAX_Z],
        planned_support_count=len(plans),plans=plans,remaining_failures=failures,
        input_files=[dict(path=p,sha256=digest) for p,digest in sorted(INPUT_HASHES.items())],
        original_v5_sha256=sha(ROOT/'cache/original_v5.npz'),
        method='Continuous conservative swept-solid bounds against complete original model, all nominal stumps, fixed micro-braces and all not-yet-removed supports. Independent fragments remain obstacles during each fragment removal. Final upward exit is checked where applicable.',
        numerical_positive_intersection_tolerance_mm3=TOL,
        limits=['No physical force, breaking, cure or resin-strength test.',
                'Cut locations and kerfs are specified. Full cutter bodies, handles, fingers and actual access force are not certified.',
                'Nominal cut planes define this removal state; maximum cut stumps are checked separately for rotation/internal service.'],
        elapsed_seconds=time.monotonic()-started)
    target=ROOT/'reports/v6_external_removal_validation.json';target.write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
