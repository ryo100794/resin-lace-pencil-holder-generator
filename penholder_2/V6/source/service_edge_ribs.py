"""Fine permanent meridian ribs beside the bottom service openings.

All coordinates are millimetres in sphere-local axes.  The two independent
ribs meet the X = +/-1.05 mm service opening edges after the original
openings are applied.  Their lower ends are embedded in the existing
bottom spherical band, rather than suspended below its first build layers.
This module does not change inter-layer bearings or temporary supports.
"""

from __future__ import annotations

import math

import manifold3d as mf
import numpy as np


def build_service_edge_ribs(radius: float, *, x_offset: float = 1.20,
                            tube_radius: float = 0.18,
                            radial_inset: float = 0.24,
                            upper_radial_inset: float = 0.02,
                            transition_angle_degrees: float = 20.,
                            top_y: float = -3.25,
                            angle_step_degrees: float = 2.0,
                            cross_segments: int = 16,
                            service_half_width: float = 1.05):
    """Return ``(two_ribs_manifold, metadata)`` for one spherical layer.

    Apply only to layers receiving service-window cuts.  Union the returned
    ribs with that layer, then apply the existing bore and service cuts.
    The default section is 0.36 mm across.  Its lower end is embedded in
    the original skin; the upper run is slightly raised so it supports the
    outermost shoulder material as soon as that shoulder begins printing.
    """
    radius = float(radius)
    center_radius = radius - float(upper_radial_inset)
    if radial_inset <= tube_radius:
        raise ValueError('The whole rib must remain within the outer sphere')
    if x_offset - tube_radius >= service_half_width:
        raise ValueError('The rib must reach the actual service cut edge')
    if cross_segments < 8 or cross_segments % 4:
        raise ValueError('Use a cross-section segment count divisible by four')
    meridian_radius = math.sqrt(center_radius**2 - x_offset**2)
    if not -meridian_radius < top_y < 0:
        raise ValueError('top_y must lie above the meridian bottom and below zero')
    last_angle = math.acos(-top_y / meridian_radius)
    steps = max(2, math.ceil(last_angle / math.radians(angle_step_degrees)))
    angles = np.linspace(0., last_angle, steps + 1)
    progress = np.minimum(1., np.sin(angles) / math.sin(math.radians(transition_angle_degrees)))**2
    center_radii = radius - radial_inset + (radial_inset - upper_radial_inset) * progress
    meridian_radii = np.sqrt(center_radii**2 - x_offset**2)
    phases = np.arange(cross_segments) * (2 * math.pi / cross_segments)
    ribs, records = [], []
    for side in (-1, 1):
        x = side * x_offset
        ct, st = np.cos(angles), np.sin(angles)
        centers = np.column_stack((np.full(len(angles), x),
                                   -meridian_radii * ct, meridian_radii * st))
        radial = np.column_stack((np.zeros(len(angles)), -ct, st))
        vertices = (centers[:, None, :]
                    + tube_radius * np.cos(phases)[None, :, None]
                    * np.array([1., 0., 0.])[None, None, :]
                    + tube_radius * np.sin(phases)[None, :, None]
                    * radial[:, None, :]).reshape((-1, 3))
        faces = []
        for i in range(steps):
            for j in range(cross_segments):
                a = i * cross_segments + j
                b = i * cross_segments + (j + 1) % cross_segments
                c = (i + 1) * cross_segments + (j + 1) % cross_segments
                d = (i + 1) * cross_segments + j
                faces.extend(((a, b, c), (a, c, d)))
        for j in range(1, cross_segments - 1):
            faces.append((0, j + 1, j))
            end = steps * cross_segments
            faces.append((end, end + j, end + j + 1))
        faces = np.asarray(faces, dtype=np.uint32)
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0:
            faces = np.ascontiguousarray(faces[:, ::-1])
        rib = mf.Manifold(mf.Mesh64(vertices, faces))
        if rib.status() != mf.Error.NoError or len(rib.decompose()) != 1:
            raise RuntimeError(f'Invalid service-edge rib: {rib.status()}')
        ribs.append(rib)
        records.append(dict(side=side, x_center_mm=x,
                            centerline_start_mm=centers[0].tolist(),
                            centerline_end_mm=centers[-1].tolist(),
                            bounds_mm=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                            maximum_vertex_spherical_radius_mm=float(np.linalg.norm(vertices, axis=1).max()),
                            volume_mm3=float(rib.volume()), triangles=int(rib.num_tri())))
    result = mf.Manifold.batch_boolean(ribs, mf.OpType.Add)
    metadata = dict(kind='permanent_service_edge_meridian_ribs', count=2,
                    layer_radius_mm=radius, tube_radius_mm=tube_radius,
                    x_offset_mm=x_offset, centerline_radial_inset_mm=radial_inset,
                    upper_centerline_radial_inset_mm=upper_radial_inset,
                    lower_transition_angle_degrees=transition_angle_degrees,
                    top_centerline_y_mm=top_y,
                    service_half_width_mm=service_half_width,
                    untrimmed_overlap_into_service_cut_x_mm=service_half_width - x_offset + tube_radius,
                    requires_original_service_cuts=True,
                    maximum_spherical_radius_bound_mm=center_radius + tube_radius,
                    role='Connect the two service-opening shoulders to the earlier bottom spherical band',
                    ribs=records)
    return result, metadata
