#!/usr/bin/env python3
"""Independent, bounded-memory audit of the actual exported NPZ surface.

Only numpy and scipy are required. The audit inspects the provided triangle
mesh without repairing, welding, filtering, or changing it. Units are mm.
It verifies topology, component count, winding and exact projected clearance
above a selected height. It does not certify resin strength or print settings.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def edge_nearest_xy(a, b):
    d = b[:, :2] - a[:, :2]
    dd = np.einsum('ij,ij->i', d, d)
    t = -np.einsum('ij,ij->i', a[:, :2], d) / np.maximum(dd, 1e-300)
    t = np.clip(t, 0, 1)
    p = a + t[:, None] * (b-a)
    return np.einsum('ij,ij->i', p[:, :2], p[:, :2]), p


def projected_minimum(tris):
    if len(tris) == 0:
        return np.inf, None
    best, point = np.inf, None
    cross_sign = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        a, b = tris[:, i], tris[:, j]
        values, pp = edge_nearest_xy(a, b)
        at = int(np.argmin(values))
        if values[at] < best:
            best, point = float(values[at]), pp[at]
        cross_sign.append(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    cross_sign = np.column_stack(cross_sign)
    # With nonzero projected area, equal signs indicate that the axis passes
    # through the projected triangle. Degenerate projections use edges above.
    area2 = np.sum(cross_sign, axis=1)
    inside = (np.abs(area2) > 1e-16) & (
        np.all(cross_sign >= 0, axis=1) | np.all(cross_sign <= 0, axis=1))
    if inside.any():
        candidates = np.flatnonzero(inside)
        weights = cross_sign[candidates][:, [1, 2, 0]] / area2[candidates, None]
        pp = np.einsum('ij,ijk->ik', weights, tris[candidates])
        # Collinear sliver projections can falsely satisfy a floating-point
        # sign test. A reconstructed intersection must actually lie on-axis.
        residual = np.linalg.norm(pp[:, :2], axis=1)
        valid = residual < 1e-8
        if valid.any():
            best, point = 0.0, pp[np.flatnonzero(valid)[0]]
    return best, point


def clip_at_z(tri, z):
    out = []
    for i in range(3):
        a, b = tri[i-1], tri[i]
        ia, ib = a[2] >= z, b[2] >= z
        if ia != ib:
            p = a+(b-a)*((z-a[2])/(b[2]-a[2]))
            p[2] = z
            out.append(p)
        if ib:
            out.append(b)
    if len(out) < 3:
        return []
    return [[out[0], out[i], out[i+1]] for i in range(1, len(out)-1)]


def sha256_file(path):
    hh = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            hh.update(chunk)
    return hh.hexdigest()


def audit(path, z_clear=4., expected_components=1):
    start = time.time()
    with np.load(path) as data:
        v = np.asarray(data['vertices'], dtype=np.float64)
        f = np.asarray(data['faces'], dtype=np.int32)
    nv, nf = len(v), len(f)
    report = dict(mesh=str(path.resolve()), unit='mm',
                  vertices=nv, triangles=nf, sha256=sha256_file(path),
                  checked_without_modification=True)
    finite = bool(np.isfinite(v).all())
    valid_indices = bool(f.min() >= 0 and f.max() < nv)
    report.update(finite_coordinates=finite, valid_indices=valid_indices)
    if not finite or not valid_indices:
        report['passed'] = False
        return report
    report['bounds_mm'] = [v.min(axis=0).tolist(), v.max(axis=0).tolist()]
    report['dimensions_mm'] = np.ptp(v, axis=0).tolist()

    # Two directed sides per triangle span all of its vertices. scipy treats
    # the graph as undirected; no repair or coordinate welding is performed.
    row = np.concatenate((f[:, 0], f[:, 0]))
    col = np.concatenate((f[:, 1], f[:, 2]))
    graph = sparse.coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)),
                              shape=(nv, nv)).tocsr()
    del row, col
    ncomp, labels = connected_components(graph, directed=False)
    del graph
    face_labels = labels[f[:, 0]]
    vc = np.bincount(labels, minlength=ncomp)
    fc = np.bincount(face_labels, minlength=ncomp)
    volumes = np.zeros(ncomp)
    repeated = 0
    zero_area = 0
    tiny_area = 0
    min_area = np.inf
    best_radial2 = np.inf
    best_position = None
    for offset in range(0, nf, 150000):
        ff = f[offset:offset+150000]
        t = v[ff]
        areas = .5*np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
        repeated += int(np.count_nonzero((ff[:, 0] == ff[:, 1]) |
                                        (ff[:, 1] == ff[:, 2]) | (ff[:, 2] == ff[:, 0])))
        zero_area += int(np.count_nonzero(areas == 0))
        tiny_area += int(np.count_nonzero(areas <= 1e-14))
        min_area = min(min_area, float(areas.min()))
        signed = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2]))/6.
        volumes += np.bincount(face_labels[offset:offset+len(t)], weights=signed, minlength=ncomp)
        eligible = t[:, :, 2] >= z_clear
        r2, point = projected_minimum(t[np.all(eligible, axis=1)])
        if r2 < best_radial2:
            best_radial2, best_position = r2, point
        crossing = t[np.any(eligible, axis=1) & ~np.all(eligible, axis=1)]
        if len(crossing):
            clipped = np.asarray([p for tri in crossing for p in clip_at_z(tri, z_clear)])
            r2, point = projected_minimum(clipped)
            if r2 < best_radial2:
                best_radial2, best_position = r2, point
    del face_labels
    components = []
    largest = np.argsort(-np.abs(volumes))[:30]
    for c in largest:
        vs = v[labels == c]
        components.append(dict(index=int(c), vertices=int(vc[c]), triangles=int(fc[c]),
            signed_volume_mm3=float(volumes[c]),
            bounds_mm=[vs.min(axis=0).tolist(), vs.max(axis=0).tolist()]))
    del labels
    report.update(component_count=int(ncomp), expected_components=int(expected_components),
        largest_components=components, all_component_volumes_positive=bool(np.all(volumes > 0)),
        positive_volume_surface_components=int(np.count_nonzero(volumes > 0)),
        negative_volume_surface_components=int(np.count_nonzero(volumes < 0)),
        total_negative_surface_volume_mm3=float(volumes[volumes < 0].sum()),
        total_signed_volume_mm3=float(volumes.sum()), repeated_index_triangles=repeated,
        zero_area_triangles=zero_area, triangles_at_or_below_area_tolerance=tiny_area,
        area_tolerance_mm2=1e-14, smallest_triangle_area_mm2=min_area,
        clearance_start_z_mm=z_clear,
        minimum_radial_distance_mm=float(np.sqrt(best_radial2)),
        guaranteed_axis_centered_clear_diameter_mm=float(2*np.sqrt(best_radial2)),
        nearest_axis_position_mm=best_position.tolist() if best_position is not None else None)

    # Encode each directed edge in uint64, then sort in place. The low bit
    # stores orientation. A closed, consistently wound mesh must have exactly
    # two occurrences with opposite orientation for every undirected edge.
    codes = np.empty(nf*3, dtype=np.uint64)
    for k, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
        va, vb = f[:, a].astype(np.uint64), f[:, b].astype(np.uint64)
        codes[k*nf:(k+1)*nf] = (np.minimum(va, vb)*nv+np.maximum(va, vb))*2+(va > vb)
    del va, vb
    codes.sort()
    raw = codes >> 1
    starts = np.r_[0, np.flatnonzero(np.diff(raw))+1]
    counts = np.diff(np.r_[starts, len(raw)])
    del raw
    closed = bool(np.all(counts == 2))
    double_starts = starts[counts == 2]
    same_orientation = int(np.count_nonzero((codes[double_starts] & 1) == (codes[double_starts+1] & 1)))
    report.update(unique_edges=int(len(starts)), boundary_edges=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        inconsistent_winding_edges=same_orientation, watertight_topology=closed,
        consistent_winding=bool(same_orientation == 0))
    report['passed'] = bool(closed and same_orientation == 0 and zero_area == 0 and repeated == 0
                            and ncomp == expected_components and np.all(volumes > 0))
    report['scope_limits'] = [
        'Topological connectivity is evaluated by shared vertex indices in the supplied mesh.',
        'Negative-volume closed surface components may be cavity boundaries, not disconnected material.',
        'Radial clearance is exact for the supplied triangles, clipped at clearance_start_z_mm.',
        'The audit does not perform a complete triangle self-intersection test.',
        'No strength, resin shrinkage, support, or exposure validation is implied.']
    report['elapsed_seconds'] = time.time()-start
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--z-clear', type=float, default=4.)
    parser.add_argument('--expected-components', type=int, default=1)
    args = parser.parse_args()
    result = audit(args.mesh, args.z_clear, args.expected_components)
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output+'\n')
    print(output, flush=True)
