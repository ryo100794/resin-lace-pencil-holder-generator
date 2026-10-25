"""Locally flatten unsupported downward tips in a permanent spherical layer.

The two simultaneous bottom receiving lands are valid roots.  Every other
positive-volume prefix component must have reached one of those roots.  A
small isolated ornament tip is removed only up to its measured joining
height.  This is a geometric construction aid, not a strength or slicer
certificate; the completed assembly needs its independent full-height audit.
"""
from __future__ import annotations

import math
import time

import manifold3d as mf
import numpy as np


VOLUME_EPSILON_MM3 = 1e-12
ROOT_LEVEL_TOLERANCE_MM = 1e-7
JOIN_RESOLUTION_MM = 2e-7


def _volume(part):
    return 0. if part.is_empty() else max(0., float(part.volume()))


def _overlap(a, b):
    if a.is_empty() or b.is_empty():
        return 0.
    aa, bb = np.asarray(a.bounding_box()), np.asarray(b.bounding_box())
    if np.any(aa[3:] < bb[:3]) or np.any(bb[3:] < aa[:3]):
        return 0.
    return _volume(a ^ b)


def _prefix(part, height):
    return part.trim_by_plane((0., -1., 0.), -float(height))


def _components(part, height):
    return [piece for piece in _prefix(part, height).decompose()
            if _volume(piece) > VOLUME_EPSILON_MM3]


def _rooted(part, root_y):
    return part.bounding_box()[1] <= root_y + ROOT_LEVEL_TOLERANCE_MM


def _minimum_levels(solid):
    mesh = solid.to_mesh64()
    vertices = np.asarray(mesh.vert_properties[:, :3], dtype=float)
    faces = np.asarray(mesh.tri_verts, dtype=np.uint32)
    levels = vertices[:, 1]
    neighbors = np.full(len(levels), np.inf)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = faces[:, first], faces[:, second]
        np.minimum.at(neighbors, a, levels[b])
        np.minimum.at(neighbors, b, levels[a])
    selected = levels <= neighbors + 1e-10
    return np.unique(np.round(levels[selected], 10))


def _find_unrooted(solid, root_y, *, grid_step=None):
    low, high = solid.bounding_box()[1], solid.bounding_box()[4]
    levels = _minimum_levels(solid)
    targets = set()
    for level in levels:
        for rise in (.0001, .001, .005, .025):
            if low < level + rise < high:
                targets.add(float(level + rise))
    if grid_step:
        start = math.floor(low/grid_step)*grid_step
        targets.update(float(y) for y in np.arange(start, high, grid_step)
                       if y > low)
    for height in sorted(targets):
        parts = _components(solid, height)
        if len(parts) == 1 and _rooted(parts[0], root_y):
            continue
        for part in parts:
            if not _rooted(part, root_y):
                return height, part
    return None


def _containing_component(solid, height, seed):
    candidates = []
    for part in _components(solid, height):
        amount = _overlap(part, seed)
        if amount > VOLUME_EPSILON_MM3:
            candidates.append((amount, part))
    if not candidates:
        raise RuntimeError('Could not track an unsupported component to its join')
    return max(candidates, key=lambda pair: pair[0])[1]


def flatten_disconnected_starts(solid, *, layer_index=None,
                               protected_parts=(), max_cuts=40,
                               max_single_removed_mm3=.50,
                               max_total_removed_fraction=.004,
                               max_tip_height_mm=.75,
                               transverse_margin_mm=.02,
                               join_margin_mm=.002,
                               verify_grid_step_mm=.05):
    """Return ``(modified_layer, records, cutters)`` in local +Y build order.

    ``protected_parts`` should contain the layer's journal, pivot and bore
    features.  Their union is removed from the geometric cutter, leaving
    the protected material intact.  Any positive removal after that relief
    is rejected.  Apply the returned, relieved cutters to the diagnostic
    shell-only mesh too, so caches describe consistent permanent geometry.
    """
    if solid.status() != mf.Error.NoError or len(solid.decompose()) != 1:
        raise ValueError('The permanent input layer must be one valid solid')
    started = time.monotonic()
    original = solid
    original_volume = _volume(solid)
    root_y = float(solid.bounding_box()[1])
    protected_parts = list(protected_parts)
    protected_union = (mf.Manifold.batch_boolean(protected_parts, mf.OpType.Add)
                       if protected_parts else mf.Manifold())
    records, cutters = [], []
    while True:
        found = _find_unrooted(solid, root_y)
        if found is None:
            # This denser check also catches any minima not selected by
            # boundary adjacency.  It is separate from the assembly audit.
            found = _find_unrooted(solid, root_y, grid_step=verify_grid_step_mm)
        if found is None:
            break
        if len(records) >= max_cuts:
            raise RuntimeError('Too many local tip corrections; manual design review required')
        detected_y, seed = found
        low = detected_y
        high = min(float(solid.bounding_box()[4]), detected_y + max_tip_height_mm)
        if not _rooted(_containing_component(solid, high, seed), root_y):
            raise RuntimeError(f'Unsupported feature does not join within {max_tip_height_mm} mm; '
                               f'layer={layer_index}, seed={seed.bounding_box()}')
        while high-low > JOIN_RESOLUTION_MM:
            middle = (low+high)/2
            if _rooted(_containing_component(solid, middle, seed), root_y):
                high = middle
            else:
                low = middle
        before_join = _containing_component(solid, low, seed)
        if _rooted(before_join, root_y):
            raise RuntimeError('Unstable local joining-height classification')
        bounds = np.asarray(before_join.bounding_box(), dtype=float)
        cut_min = bounds[:3] - [transverse_margin_mm, .01, transverse_margin_mm]
        cut_max = bounds[3:] + [transverse_margin_mm, 0., transverse_margin_mm]
        cut_max[1] = high + join_margin_mm
        if cut_min[1] <= root_y+.08:
            raise RuntimeError('Local trim would modify the bottom receiving-land transition')
        raw_cutter = mf.Manifold.cube(tuple(cut_max-cut_min)).translate(tuple(cut_min))
        raw_removed_volume = _overlap(solid, raw_cutter)
        cutter = raw_cutter-protected_union if protected_parts else raw_cutter
        removed = solid ^ cutter
        removed_volume = _volume(removed)
        if not 0 < removed_volume <= max_single_removed_mm3:
            raise RuntimeError(f'Local trim volume requires review: {removed_volume} mm3')
        protected_overlap = [_overlap(removed, part) for part in protected_parts]
        if protected_overlap and max(protected_overlap) > 1e-10:
            raise RuntimeError(f'Local trim reaches a protected journal feature: '
                               f'layer={layer_index}, overlaps={protected_overlap}, '
                               f'bounds={list(cut_min)+list(cut_max)}')
        candidate = solid - cutter
        if _overlap(candidate, seed) > VOLUME_EPSILON_MM3:
            raise RuntimeError('The unsupported start includes protected journal material; '
                               'manual design review required')
        if candidate.status() != mf.Error.NoError or len(candidate.decompose()) != 1:
            raise RuntimeError('Local trim disconnects the completed permanent layer')
        total_removed = _volume(original-candidate)
        if total_removed > original_volume * max_total_removed_fraction:
            raise RuntimeError(f'Total local removal requires review: {total_removed} mm3')
        record = dict(layer_index=layer_index,
                      role='Remove a downward ornamental tip until its first retained material is root-connected',
                      original_tip_first_y_mm=float(seed.bounding_box()[1]),
                      detection_height_y_mm=float(detected_y),
                      disconnected_seed_volume_mm3=_volume(seed),
                      measured_join_height_bracket_mm=[float(low), float(high)],
                      flatten_height_mm=float(cut_max[1]-seed.bounding_box()[1]),
                      box_min_mm=cut_min.tolist(), box_max_mm=cut_max.tolist(),
                      actual_removed_shell_volume_mm3=removed_volume,
                      raw_box_removal_before_protected_relief_mm3=raw_removed_volume,
                      protected_material_preserved_from_raw_box_mm3=max(0., raw_removed_volume-removed_volume),
                      protected_feature_overlap_mm3=protected_overlap,
                      join_margin_mm=float(join_margin_mm),
                      transverse_margin_mm=float(transverse_margin_mm))
        records.append(record)
        cutters.append(cutter)
        solid = candidate
    total_removed = _volume(original-solid)
    if records:
        for record in records:
            record['layer_total_removed_mm3'] = total_removed
            record['layer_total_removed_fraction'] = total_removed/original_volume
            record['layer_connectivity_sampling_grid_mm'] = float(verify_grid_step_mm)
            record['layer_connectivity_sampling_passed'] = True
            record['layer_correction_seconds'] = time.monotonic()-started
    return solid, records, cutters
