"""Complete two-way triangle-surface bound for Mesh64 import retriangulation.

No mesh is edited. Exact unchanged triangles have zero error. Every changed
triangle is covered either by one actual target triangle, or by a union of
parallel target-triangle projections with an explicitly bounded planar pad.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh.triangles import closest_point

ROW=np.dtype([('a',np.uint32),('b',np.uint32),('c',np.uint32)])

def canonical(f):
    where=np.argmin(f,axis=1)
    out=np.ascontiguousarray(np.take_along_axis(f,(where[:,None]+np.arange(3)[None,:])%3,axis=1),dtype=np.uint32)
    return out.view(ROW).ravel()

def cover_triangle(query,targets,budget=1e-8):
    if not len(targets):return dict(passed=False,reason='No actual target triangle candidate')
    # Distance to a fixed convex triangle is convex. Bounding it at every
    # query vertex bounds every point of the complete query triangle.
    points=np.tile(query,(len(targets),1));tri=np.repeat(targets,3,axis=0)
    distances=np.linalg.norm(closest_point(tri,points)-points,axis=1).reshape(-1,3)
    one=float(distances.max(axis=1).min())
    if one<=budget:
        return dict(passed=True,bound_mm=one+2e-12,method='Entire triangle to one actual convex target triangle')
    edge=query[1]-query[0];cross=np.cross(edge,query[2]-query[0]);norm=float(np.linalg.norm(cross))
    if norm<1e-26:return dict(passed=False,reason='Degenerate source triangle has no direct convex covering target',one_triangle_bound_mm=one)
    normal=cross/norm;u=edge/np.linalg.norm(edge);vv=np.cross(normal,u)
    relative=targets-query[0]
    plane_distance=np.abs(relative@normal).max(axis=1)
    normals=np.cross(targets[:,1]-targets[:,0],targets[:,2]-targets[:,0])
    aligned=(normals@normal)>0
    usable=(plane_distance<=budget)&aligned
    if not np.any(usable):return dict(passed=False,reason='No suitably coplanar oriented target triangles',one_triangle_bound_mm=one)
    actual=relative[usable];xy=np.stack((actual@u,actual@vv),axis=2)
    polygons=[Polygon(t)for t in xy if abs(np.cross(t[1]-t[0],t[2]-t[0]))>1e-28]
    if not polygons:return dict(passed=False,reason='No nondegenerate projected covering facets')
    cover=unary_union(polygons)
    qrel=query-query[0];footprint=Polygon(np.column_stack((qrel@u,qrel@vv)))
    plane_bound=float(plane_distance[usable].max())
    for pad in [0.,1e-12,1e-10,1e-8]:
        # Shapely's polygonal circular buffer is inscribed in the exact
        # radius-pad offset; coverage therefore proves a distance <=pad.
        expanded=cover if pad==0 else cover.buffer(pad,resolution=8)
        if expanded.covers(footprint):
            return dict(passed=True,bound_mm=math.hypot(pad,plane_bound)+2e-12,
                method='Complete oriented projected-target-triangle union covers query',
                planar_pad_mm=pad,maximum_target_plane_distance_mm=plane_bound,covering_triangle_count=len(polygons))
    return dict(passed=False,reason='Complete projected coverage could not be proved',
        one_triangle_bound_mm=one,projected_uncovered_area_mm2=float(footprint.difference(cover).area))

def directional(changed,source_coords,target_faces,target_coords,label):
    if not len(changed):return dict(passed=True,changed_triangles=0,bound_mm=0.,rows=[])
    actual=source_coords[changed]
    target_unique=np.unique(target_faces)
    tree=cKDTree(target_coords[target_unique])
    _,nearest=tree.query(actual.reshape(-1,3),k=min(12,len(target_unique)),workers=1)
    neighborhood=np.unique(target_unique[np.asarray(nearest).reshape(-1)])
    pools=[]
    for start in range(0,len(target_faces),65536):
        f=target_faces[start:start+65536]
        mask=np.isin(f,neighborhood).any(axis=1)
        if np.any(mask):pools.append(f[mask])
    pool=target_coords[np.vstack(pools)]
    low=pool.min(axis=1);high=pool.max(axis=1);rows=[];maximum=0.
    for i,query in enumerate(actual):
        selected=np.all(high>=query.min(axis=0)-1e-7,axis=1)&np.all(low<=query.max(axis=0)+1e-7,axis=1)
        result=cover_triangle(query,pool[selected])
        result.update(index=i,source_triangle_vertices_mm=query.tolist())
        rows.append(result)
        if result['passed']:maximum=max(maximum,result['bound_mm'])
        else:print('Surface cover failed',label,i,result['reason'],flush=True)
    return dict(passed=all(r['passed']for r in rows),changed_triangles=len(changed),bound_mm=maximum,rows=rows)

def prove(v0,f0,v1,f1):
    shared,old_inverse=np.unique(v0,axis=0,return_inverse=True)
    tree=cKDTree(shared);distance,new_inverse=tree.query(v1,workers=1)
    if np.any(distance):
        shared,inverse=np.unique(np.vstack((v0,v1)),axis=0,return_inverse=True)
        old_inverse=inverse[:len(v0)];new_inverse=inverse[len(v0):]
    old_faces=old_inverse[f0];new_faces=new_inverse[f1]
    old=canonical(old_faces);new=canonical(new_faces)
    missing=np.setdiff1d(old,new).view(np.uint32).reshape(-1,3)
    added=np.setdiff1d(new,old).view(np.uint32).reshape(-1,3)
    print('Import changed oriented triangle sets:',len(missing),'old;',len(added),'new',flush=True)
    a=directional(missing,shared,new_faces,shared,'old-to-new')
    b=directional(added,shared,old_faces,shared,'new-to-old')
    return dict(passed=a['passed']and b['passed'],
        complete_bidirectional_surface_distance_upper_bound_mm=max(a['bound_mm'],b['bound_mm']),
        old_to_new=a,new_to_old=b,input_triangle_count=len(f0),output_triangle_count=len(f1),
        exact_unchanged_triangle_coordinate_set_checked=True,
        output_vertices_subset_of_input=bool(np.all(distance==0)),
        no_mesh_mutation_or_component_filtering=True,
        numerical_proof_padding_mm=2e-12,
        proof='Every exact unchanged oriented triangle is retained geometrically. Each changed triangle is covered in its entirety by actual opposite-mesh triangles. Fixed-triangle distance convexity or complete planar polygon coverage, with explicit perpendicular/parallel distance bounds, proves both directed surface bounds; no point-sampling inference is used.')
