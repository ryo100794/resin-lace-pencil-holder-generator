#!/usr/bin/env python3
"""Render precise explanatory views of the nested-sphere support mechanism.

All displayed solids are supplied modelling meshes.  Explanatory colors and
display clipping planes distinguish permanent bearings from removable supports.
Clipping, camera movement and color never change any source or print mesh.

The input JSON is the actual component and endpoint metadata written by
``sphere_geometry.py``.  Local +Z faces outward, local +Y is up and local +X is
tangential.  ``render_model.py`` supplies the shared EGL studio and verified
baseline-JPEG writer.  The renderer never changes the supplied source files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from render_model import (Studio, arrays_to_mesh, atomic_image, bg_rgb, font, load_mesh,
                          numpy_to_vtk, revision_details, sha256, vtk, vtk_to_numpy, IVORY)


ROOT = Path(__file__).resolve().parent.parent
PALETTE = {
    "shell": IVORY,
    "bearing": (0.29, 0.68, 0.75),
    "pivot": (0.29, 0.68, 0.75),
    "wire": (0.94, 0.46, 0.29),
    "neck": (1.00, 0.72, 0.36),
    "grip": (0.72, 0.60, 0.88),
}
TEMPORARY_ROLES = ("wire", "neck", "grip")
INK = (236, 231, 221)
QUIET = (194, 188, 176)
PANEL = (54, 51, 46)


def rgb(role):
    return tuple(round(255 * channel) for channel in PALETTE[role])


def world(local):
    """Rigid coordinate permutation only, preserving all metric lengths."""
    local = np.asarray(local, dtype=float)
    return local[..., [2, 0, 1]]


def view_angles(local_direction):
    d = world(local_direction)
    d /= np.linalg.norm(d)
    return {"azimuth": math.degrees(math.atan2(d[1], d[0])),
            "elevation": math.degrees(math.asin(float(d[2])))}


@dataclass
class Part:
    name: str
    role: str
    path: Path
    mesh: object
    layer: int | None = None
    lighting: str = "crease"
    surface_partition: bool = False


def set_area_weighted_normals(mesh):
    """Give very narrow curved-union triangles proportionate lighting weight."""
    vertices = np.asarray(vtk_to_numpy(mesh.GetPoints().GetData()), dtype=np.float64)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    values = numpy_to_vtk(np.ascontiguousarray(normals, dtype=np.float32), deep=True)
    values.SetName("AreaWeightedLightingNormals")
    mesh.GetPointData().SetNormals(values)


class MechanismStudio(Studio):
    """A studio with independently colored, optionally clipped actual parts."""

    def __init__(self, parts, ao=True):
        append = vtk.vtkAppendPolyData()
        for part in parts:
            append.AddInputData(part.mesh)
        append.Update()
        combined = vtk.vtkPolyData()
        combined.ShallowCopy(append.GetOutput())
        super().__init__(combined, ao=ao)
        self.actor.SetVisibility(False)
        self.parts = parts
        self.part_actors = []
        self.part_normals = []
        self.section_filters = []
        for part in parts:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(part.mesh)
            # Unioned curved-wire facets have some near-tangent, very narrow
            # faces.  Splitting their normals creates false dotted seams.
            # Continuous lighting normals remove those rendering artifacts;
            # the actual triangle connectivity and all vertices are retained.
            if part.lighting == "area_weighted":
                normals.SplittingOff()
            else:
                normals.SplittingOn()
            normals.SetFeatureAngle(35)
            normals.ConsistencyOn()
            if part.surface_partition:
                # Exterior-face subsets inherit the source's correct winding.
                # They are open display patches, not independently closed solids.
                normals.AutoOrientNormalsOff()
            else:
                normals.AutoOrientNormalsOn()
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            if part.lighting == "area_weighted":
                # The source contains many sliver facets at exact cylinder /
                # sphere unions.  Weighting each face by its actual area avoids
                # false ripples from giving a tiny facet a whole face's weight.
                set_area_weighted_normals(part.mesh)
                normals.ComputePointNormalsOff()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            # The neck colors are exact disjoint Boolean display regions.
            # No depth bias or geometry inflation is needed to show them.
            mapper.SetResolveCoincidentTopologyToOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*PALETTE[part.role])
            prop.SetAmbient(0.26 if part.role != "shell" else 0.22)
            prop.SetDiffuse(0.71)
            prop.SetSpecular(0.065)
            prop.SetSpecularPower(24)
            prop.SetInterpolationToPhong()
            prop.BackfaceCullingOff()
            self.renderer.AddActor(actor)
            self.part_actors.append(actor)
            self.part_normals.append(normals)

    def section(self, normal_local=None, origin_local=(0, 0, 0), roles=("shell",), cap=False):
        self.section_filters = []
        for part, actor, original_normals in zip(self.parts, self.part_actors, self.part_normals):
            mapper = actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
            mapper.SetInputConnection(original_normals.GetOutputPort())
            if normal_local is not None and part.role in roles:
                plane = vtk.vtkPlane()
                plane.SetNormal(*world(normal_local))
                plane.SetOrigin(*world(origin_local))
                if cap:
                    # The display section of a solid pin must remain visibly
                    # solid.  Exact plane-intersection faces close the drawing
                    # only; the uncut source and print meshes are untouched.
                    planes = vtk.vtkPlaneCollection()
                    planes.AddItem(plane)
                    clipped = vtk.vtkClipClosedSurface()
                    clipped.SetInputData(part.mesh)
                    clipped.SetClippingPlanes(planes)
                    clipped.GenerateFacesOn()
                    normals = vtk.vtkPolyDataNormals()
                    normals.SetInputConnection(clipped.GetOutputPort())
                    normals.SplittingOn()
                    normals.SetFeatureAngle(35)
                    normals.ConsistencyOn()
                    normals.AutoOrientNormalsOn()
                    normals.ComputePointNormalsOn()
                    mapper.SetInputConnection(normals.GetOutputPort())
                    self.section_filters.extend([planes, clipped, normals])
                else:
                    mapper.AddClippingPlane(plane)

    def show_roles(self, roles=None, layers=None):
        for part, actor in zip(self.parts, self.part_actors):
            visible = roles is None or part.role in roles
            if layers is not None and part.layer is not None:
                visible = visible and part.layer in layers
            actor.SetVisibility(visible)

    def highlight_wire(self, name_prefix=None):
        """Keep every solid opaque while distinguishing supplied support(s)."""
        prefixes = [name_prefix] if isinstance(name_prefix, str) else name_prefix
        for part, actor in zip(self.parts, self.part_actors):
            color = np.asarray(PALETTE[part.role], dtype=float)
            if (prefixes is not None and part.role in TEMPORARY_ROLES
                    and not any(part.name == prefix or part.name.startswith(prefix + "_")
                                for prefix in prefixes)):
                color *= 0.47
            actor.GetProperty().SetColor(*color)

    def show_prefixes(self, prefixes):
        """Isolate an explicitly labelled part detail, without moving geometry."""
        for part, actor in zip(self.parts, self.part_actors):
            actor.SetVisibility(any(part.name == prefix or part.name.startswith(prefix + "_")
                                    for prefix in prefixes))

    def project(self, local_point):
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        coordinate.SetValue(*world(local_point))
        x, y = coordinate.GetComputedDoubleDisplayValue(self.renderer)
        return float(x), float(self.window.GetSize()[1] - y)


def leader(image, target, label, anchor, color=INK, right=False):
    """Draw one leader to a projected, supplied coordinate of the actual part."""
    draw = ImageDraw.Draw(image)
    label_font = font(20)
    x, y = anchor
    bbox = draw.textbbox((0, 0), label, font=label_font)
    width = bbox[2] - bbox[0]
    text_x = x - width if right else x
    center_y = y + 13
    line_x = text_x - 9 if right else text_x + width + 9
    elbow = ((line_x + target[0]) * 0.5, center_y)
    draw.line([target, elbow, (line_x, center_y)], fill=color, width=2)
    radius = 3
    draw.ellipse((target[0] - radius, target[1] - radius,
                  target[0] + radius, target[1] + radius), fill=color)
    draw.rounded_rectangle((text_x - 5, y - 3, text_x + width + 5, y + 28),
                           radius=3, fill=PANEL)
    draw.text((text_x, y), label, font=label_font, fill=color)


def dimension_marker(image, p0, p1, name, label_offset=(16, -20)):
    """Mark the projected endpoints of one exact metric gap from metadata."""
    draw = ImageDraw.Draw(image)
    p0, p1 = np.asarray(p0), np.asarray(p1)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    if length < 3:
        return
    along = vector / length
    across = np.array([-along[1], along[0]])
    draw.line([tuple(p0), tuple(p1)], fill=INK, width=2)
    for point, direction in [(p0, along), (p1, -along)]:
        draw.line([tuple(point - across * 7), tuple(point + across * 7)], fill=INK, width=2)
        back = point + direction * min(8, length * 0.24)
        draw.polygon([tuple(point), tuple(back + across * 4), tuple(back - across * 4)], fill=INK)
    middle = (p0 + p1) * 0.5 + np.asarray(label_offset)
    x, y = tuple(middle)
    draw.rounded_rectangle((x - 8, y - 4, x + 19, y + 28), radius=4, fill=PANEL)
    draw.text((x, y), name, font=font(20, bold=True), fill=INK)


def legend(draw, origin, entries, step=250, fontsize=18):
    x, y = origin
    for index, (role, label) in enumerate(entries):
        left = x + index * step
        draw.ellipse((left, y + 3, left + 13, y + 16), fill=rgb(role))
        draw.text((left + 22, y), label, font=font(fontsize), fill=QUIET)


def fit_text(draw, xy, text, max_width, size=18, fill=QUIET):
    """Wrap factual text without shrinking to unreadable caption sizes."""
    words = text.split()
    lines, line = [], ""
    f = font(size)
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=f) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    draw.multiline_text(xy, "\n".join(lines), font=f, fill=fill, spacing=7)
    return len(lines) * (size + 7)


def first_value(mapping, names, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def find_numeric(mapping, names):
    """Read an explicit dimension from metadata without introducing defaults."""
    found = []
    if isinstance(mapping, dict):
        for name, value in mapping.items():
            if name in names and isinstance(value, (int, float)):
                found.append(float(value))
            elif isinstance(value, (dict, list)):
                answer = find_numeric(value, names)
                if answer is not None:
                    found.append(answer)
    elif isinstance(mapping, list):
        for item in mapping:
            answer = find_numeric(item, names)
            if answer is not None:
                found.append(answer)
    if not found:
        return None
    if max(found) - min(found) > 1e-7:
        # Do not label a global dimension if the supplied design varies it.
        return None
    return found[0]


def file_name(entry):
    return entry["mesh_file"] if isinstance(entry, dict) else entry


def axis_point(bearing, scalar_name, axis, signed=False):
    for key in [scalar_name.replace("_axial_mm", "_center"),
                scalar_name.replace("_axial_mm", "_point")]:
        if key in bearing:
            return np.asarray(bearing[key], dtype=float)
    value = float(bearing[scalar_name])
    if not signed:
        side = bearing.get("side", 1)
        if isinstance(side, str):
            side = -1 if side.lower() in ("-", "negative", "minus", "lower") else 1
        value *= float(side)
    return np.asarray(axis) * value


def wire_kind(wire):
    return str(first_value(wire, ["kind", "support_type", "type"], "front_window"))


def section_dimensions(mapping, prefix):
    """Return only an explicitly supplied width/thickness cross-section."""
    width = find_numeric(mapping, [prefix + "_width_mm"])
    thickness = find_numeric(mapping, [prefix + "_thickness_mm"])
    if width is not None and thickness is not None:
        return width, thickness
    section = mapping.get(prefix + "_cross_section_mm")
    if isinstance(section, (list, tuple)) and len(section) == 2:
        return float(section[0]), float(section[1])
    if isinstance(section, dict) and {"width", "thickness"} <= set(section):
        return float(section["width"]), float(section["thickness"])
    return None


def release_dimensions(wire):
    rows = []
    for prefix, label in [("body", "SUPPORT BODY"), ("neck", "NECK WIDTH x THICKNESS")]:
        section = section_dimensions(wire, prefix)
        if section:
            rows.append({"label": label, "value": f"{section[0]:.2f} x {section[1]:.2f} mm"})
        else:
            aliases = (["body_diameter_mm", "wire_diameter_mm", "wire_body_diameter_mm"]
                       if prefix == "body" else ["neck_diameter_mm"])
            diameter = find_numeric(wire, aliases)
            if diameter is not None:
                rows.append({"label": "SUPPORT BODY" if prefix == "body" else "BREAKAWAY NECK",
                             "value": f"D {diameter:.2f} mm"})
    for label, names in [
        ("NECK LENGTH", ["neck_length_mm"]),
        ("NECK PLANE OFFSET", ["nominal_exposed_residue_mm"]),
    ]:
        value = find_numeric(wire, names)
        if value is not None:
            rows.append({"label": label, "value": f"{value:.2f} mm"})
    residues = [float(endpoint["maximum_residue_mm"]) for endpoint in wire["endpoints"]
                if "maximum_residue_mm" in endpoint]
    if residues:
        rows.append({"label": "DESIGN STUB LIMIT", "value": f"{max(residues):.2f} mm"})
    return rows


def derive_spec(metadata, meta_path):
    """Adapt sphere_geometry's geometric metadata into reproducible viewpoints."""
    data = metadata.get("mechanism", metadata)
    if "shell_only_meshes" not in data:
        raise ValueError("Metadata lacks the actual mechanism component meshes")
    revision = revision_details(metadata)
    parts = []
    for index, entry in enumerate(data["shell_only_meshes"]):
        parts.append({"name": f"shell_{index + 1:02d}", "role": "shell",
                      "file": file_name(entry), "layer": index})
    for index, entry in enumerate(data["bearing_feature_meshes"]):
        # Diagnostic Boolean cutters describe empty clearance volumes.  They
        # are not printed solids and must never be included as colored parts.
        if entry.get("role") not in ("socket", "pivot"):
            continue
        role = "pivot" if entry.get("role") == "pivot" else "bearing"
        parts.append({"name": f"bearing_{index:02d}_{entry.get('role', 'socket')}",
                      "role": role, "file": file_name(entry),
                      "layer": entry.get("layer_index")})
    wires = data["release_wires"]
    for index, wire in enumerate(wires):
        starter = any(term in wire_kind(wire).lower() for term in ("starter", "journal"))
        parts.append({"name": f"wire_{index:02d}", "role": "wire",
                      "file": wire["mesh_file"], "layer": wire.get("host_index"),
                      "lighting": "crease" if starter and revision["number"] >= 3 else "area_weighted"})
        for neck_index, entry in enumerate(wire["neck_mesh_files"]):
            parts.append({"name": f"wire_{index:02d}_neck_{neck_index}", "role": "neck",
                          "file": file_name(entry), "layer": wire.get("host_index"),
                          "lighting": "crease" if section_dimensions(wire, "neck") else "area_weighted"})
        grips = list(wire.get("grip_mesh_files", []))
        if wire.get("grip_mesh_file"):
            grips.append(wire["grip_mesh_file"])
        for grip_index, entry in enumerate(grips):
            parts.append({"name": f"wire_{index:02d}_grip_{grip_index}", "role": "grip",
                          "file": file_name(entry), "layer": wire.get("host_index"), "lighting": "crease"})
    axis = np.asarray(data["axis_unit_vector"], dtype=float)
    axis /= np.linalg.norm(axis)
    normal = np.cross(axis, [0.0, 0.0, 1.0])
    normal /= np.linalg.norm(normal)
    outer_radius = float(first_value(metadata, ["outer_nominal_radius_mm", "outer_radius_mm"],
                                    first_value(data, ["outer_nominal_radius_mm", "outer_radius_mm"])))
    bearings = first_value(data, ["bearings", "bearing_pairs", "joints"], [])
    if not bearings:
        raise ValueError("No bearing position metadata is available for actual callouts")
    if any("bearings" in item for item in bearings):
        bearings = [bearing for item in bearings for bearing in item.get("bearings", [item])]
    host = min(int(item["host_index"]) for item in bearings)
    pair = [item for item in bearings if int(item["host_index"]) == host]
    signed = bool(data.get("bearing_axial_coordinates_are_signed", False))
    positions = [axis_point(item, "mouth_axial_mm", axis, signed) for item in pair]
    order = np.argsort([np.dot(point, axis) for point in positions])
    selected_bearing = pair[int(order[-1])]
    bearing_focus = axis_point(selected_bearing, "mouth_axial_mm", axis, signed)
    starters = [wire for wire in wires if any(term in wire_kind(wire).lower()
                                             for term in ("starter", "journal"))]
    front_wires = [wire for wire in wires if wire not in starters]
    selected_wire = min(front_wires or wires,
                        key=lambda item: item.get("sequence_index", item.get("wire_index", 0)))
    endpoints = sorted(selected_wire["endpoints"],
                       key=lambda item: np.linalg.norm(item["attachment_point"]), reverse=True)
    wire_path = Path(selected_wire["mesh_file"])
    if not wire_path.is_absolute():
        wire_path = meta_path.parent / wire_path
    with np.load(wire_path, allow_pickle=False) as source:
        verts = source["vertices"] if "vertices" in source.files else source["v"]
        desired = np.mean([endpoint["neck_center"] for endpoint in endpoints], axis=0)
        wire_point = verts[int(np.argmin(np.sum((verts - desired) ** 2, axis=1)))].tolist()
    pin_diameter = 2 * float(selected_bearing["pin_radius_mm"])
    bore_diameter = 2 * float(selected_bearing["bore_radius_mm"])
    radial_gap = (bore_diameter - pin_diameter) * 0.5
    end_gap = abs(float(selected_bearing["stop_axial_mm"]) -
                  float(selected_bearing["pin_tip_axial_mm"]))
    midpoint = (axis_point(selected_bearing, "mouth_axial_mm", axis, signed)
                + axis_point(selected_bearing, "pin_tip_axial_mm", axis, signed)) * 0.5
    radial_direction = -normal
    radial_pin = midpoint + radial_direction * float(selected_bearing["pin_radius_mm"])
    radial_bore = midpoint + radial_direction * float(selected_bearing["bore_radius_mm"])
    common_dimensions = release_dimensions(selected_wire)
    section_callouts = []
    if len(order) >= 2:
        if abs(axis[0]) > abs(axis[1]):
            labels = [(int(order[0]), "LEFT PIVOT", [24, 57], False),
                      (int(order[-1]), "RIGHT PIVOT", [676, 57], True)]
        else:
            labels = [(int(order[-1]), "OPPOSED PIVOT A", [24, 45], False),
                      (int(order[0]), "OPPOSED PIVOT B", [24, 732], False)]
        for ix, name, anchor, right in labels:
            section_callouts.append({"point_local": positions[ix].tolist(), "text": name,
                                     "anchor_px": anchor, "role": "bearing", "right": right})
    spec = {
        "revision": revision,
        "parts": parts, "outer_radius_mm": outer_radius,
        "layer_count": len(data["shell_only_meshes"]),
        "front_support_count": len(front_wires),
        "starter_support_count": len(starters),
        "support_count": len(wires),
        "neck_count": sum(len(wire["endpoints"]) for wire in wires),
        "has_grip_parts": any(part["role"] == "grip" for part in parts),
        "selected_wire": selected_wire, "release_endpoints": endpoints,
        "selected_wire_part_prefix": f"wire_{wires.index(selected_wire):02d}",
        "release_dimensions": common_dimensions,
        "overview_caption": f"Each adjacent pair of shells uses two opposed pivot bearings. "
                            f"Radial running clearance is {radial_gap:.2f} mm and axial end clearance is {end_gap:.2f} mm.",
        "views": {
            "front": {"direction_local": [0, 0, 1],
                      "callouts": [{"point_local": wire_point, "text": "REMOVABLE BRIDGE",
                                    "anchor_px": [18, 739], "role": "wire"}]},
            "section": {"direction_local": (np.array([0, 0, 1.2]) - normal * 0.70).tolist(),
                        "clip_normal_local": normal.tolist(), "callouts": section_callouts},
            "bearing": {
                "direction_local": (np.array([0, 0, 1.0]) - normal * 0.25).tolist(),
                "focus_local": midpoint.tolist(), "parallel_scale_mm": 2.4,
                "clip_normal_local": [0, 0, -1],
                "clip_roles": ["shell", "bearing", "pivot"],
                "cap_section": True,
                "dimensions": [
                    {"label": "PIVOT DIAMETER", "value": f"D {pin_diameter:.2f} mm"},
                    {"label": "SOCKET BORE", "value": f"D {bore_diameter:.2f} mm"},
                    {"label": "A  RADIAL RUNNING GAP", "value": f"{radial_gap:.2f} mm"},
                    {"label": "B  AXIAL END GAP", "value": f"{end_gap:.2f} mm"},
                    {"label": "PAIR ARRANGEMENT", "value": "180 degrees opposed"}],
                "caption": "Display section through one side of the opposed bearings. "
                           "Markers A and B refer to the outermost interface; these blue parts remain after support removal.",
                "callouts": [],
                "dimension_markers": [
                    {"name": "A", "p0_local": radial_pin.tolist(), "p1_local": radial_bore.tolist(),
                     "label_offset_px": [20, -18]},
                    {"name": "B", "p0_local": axis_point(selected_bearing, "pin_tip_axial_mm", axis, signed).tolist(),
                     "p1_local": axis_point(selected_bearing, "stop_axial_mm", axis, signed).tolist(),
                     "label_offset_px": [0, 24]}],
            },
        },
    }
    if starters:
        # Show the outer interface's actual stem and neck dimensions first;
        # that actual pair prominently.  Release sequence stays in metadata.
        starter = min(starters, key=(lambda item: (int(item["host_index"]),
                                                   item.get("sequence_index", item.get("wire_index", 0))))
                      if revision["number"] >= 3 else
                      (lambda item: item.get("sequence_index", item.get("wire_index", 0))))
        spec["selected_starter"] = starter
        spec["selected_starter_part_prefix"] = f"wire_{wires.index(starter):02d}"
        spec["starter_endpoints"] = sorted(starter["endpoints"],
                                           key=lambda item: np.linalg.norm(item["attachment_point"]),
                                           reverse=True)
        spec["starter_dimensions"] = release_dimensions(starter)
        starter_group = [wire for wire in starters
                         if (wire.get("host_index"), wire.get("moving_index"))
                         == (starter.get("host_index"), starter.get("moving_index"))]
        spec["selected_starter_group"] = starter_group
        spec["selected_starter_group_prefixes"] = [f"wire_{wires.index(wire):02d}" for wire in starter_group]
        group_vertices = []
        for wire in starter_group:
            path = Path(wire["mesh_file"])
            path = path if path.is_absolute() else meta_path.parent / path
            with np.load(path, allow_pickle=False) as arrays:
                group_vertices.append(arrays["vertices"] if "vertices" in arrays.files else arrays["v"])
        vertices = np.concatenate(group_vertices)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        focus = (lower + upper) * 0.5
        span = upper - lower
        spec["views"]["starter_pair"] = {
            "focus_local": focus.tolist(),
            "front_direction_local": [0, 0, 1],
            "oblique_direction_local": [0.38, 0.22, 1.0],
            "parallel_scale_mm": float(max(span[1] * 0.72,
                                            np.linalg.norm(span[[0, 2]]) * 0.72 * 770 / 700)),
            "front_visibility": "Complete opaque mechanism, viewed through actual service openings",
            "oblique_visibility": "Selected actual support pair only, with unchanged original coordinates",
            "shown_support_names": spec["selected_starter_group_prefixes"],
        }
    return spec


def partition_wire_colors(parts, metadata):
    """Partition actual support solids into body, break neck, and grip colors.

    Every displayed region is intersected with its complete physical support.
    The optional structural-stem mask keeps the direct orange load path whole;
    only the grip contribution outside it receives the grip color.  Straight
    starters use mutually exclusive subsets of their original exterior faces;
    this retains every original triangle exactly once, without internal color
    interfaces that could become visible through raster depth rounding.
    Legacy neck masks retain their axial extent and cover the complete physical
    cross-section.  Very small mask-only allowances prevent coincident-face
    color seams between independently exported diagnostic solids.  These
    masks are intersected with the unchanged support and are never displayed.
    """
    import manifold3d as mf

    def solid(path):
        with np.load(path, allow_pickle=False) as arrays:
            vertices = arrays["vertices"] if "vertices" in arrays.files else arrays["v"]
            faces = arrays["faces"] if "faces" in arrays.files else arrays["f"]
        return mf.Manifold(mf.Mesh64(np.asarray(vertices, dtype=np.float64),
                                     np.asarray(faces, dtype=np.uint32)))

    def display_mesh(model):
        source = model.to_mesh64()
        return arrays_to_mesh(world(source.vert_properties[:, :3]), source.tri_verts)

    def expanded_mask(model, scale):
        vertices = model.to_mesh64().vert_properties[:, :3]
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        matrix = np.eye(3) * scale
        return model.transform(np.column_stack([matrix, center - matrix @ center]))

    def original_surface_partition(part, wire, necks, grips):
        """Color a straight starter's existing faces; never add interface faces."""
        with np.load(part.path, allow_pickle=False) as arrays:
            vertices = np.asarray(arrays["vertices"], dtype=np.float64)
            faces = np.asarray(arrays["faces"], dtype=np.uint32)
        triangles = vertices[faces]
        centroids = triangles.mean(axis=1)
        assignment = np.zeros(len(faces), dtype=np.int16)
        targets = [part] + necks + grips
        # Diagnostic neck stations and source endpoint stations have identical
        # nominal axial coordinates; a 2 nm inclusion tolerance only handles
        # independently exported float32/float64 coordinate rounding.
        tolerance_mm = 2e-6
        for index, neck in enumerate(necks, 1):
            endpoint = wire["endpoints"][int(neck.name.rsplit("_", 1)[1])]
            axis = np.asarray(endpoint["inward_tangent"], dtype=float)
            axis /= np.linalg.norm(axis)
            with np.load(neck.path, allow_pickle=False) as arrays:
                axial = np.asarray(arrays["vertices"], dtype=float) @ axis
            projected = centroids @ axis
            assignment[(projected >= axial.min() - tolerance_mm)
                       & (projected <= axial.max() + tolerance_mm)] = index

        def in_convex_diagnostic(path):
            # The supplied separate grip is a triangular prism and its stem is an
            # elliptical prism.  Derive half-spaces from those supplied meshes,
            # and verify convexity before using them as color masks.
            with np.load(path, allow_pickle=False) as arrays:
                mask_vertices = np.asarray(arrays["vertices"], dtype=float)
                mask_faces = np.asarray(arrays["faces"], dtype=int)
            tri = mask_vertices[mask_faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            magnitude = np.linalg.norm(normal, axis=1)
            valid = magnitude > 1e-14
            normal = normal[valid] / magnitude[valid, None]
            origins = tri[valid, 0]
            center = mask_vertices.mean(axis=0)
            flip = np.einsum("ij,ij->i", center - origins, normal) > 0
            normal[flip] *= -1
            offset = np.einsum("ij,ij->i", origins, normal)
            if np.max(mask_vertices @ normal.T - offset) > tolerance_mm:
                raise ValueError(f"Actual starter color diagnostic is not convex: {path}")
            return np.max(centroids @ normal.T - offset, axis=1) <= tolerance_mm

        structural_stem_path = Path(wire["structural_stem_mesh_file"])
        if not structural_stem_path.is_absolute():
            structural_stem_path = part.path.parent / structural_stem_path
        inside_stem = in_convex_diagnostic(structural_stem_path)
        for index, grip in enumerate(grips, len(necks) + 1):
            visible_grip = in_convex_diagnostic(grip.path) & ~inside_stem
            assignment[(assignment == 0) & visible_grip] = index
        counts = []
        signed_volume = np.einsum("ij,ij->i", triangles[:, 0],
                                  np.cross(triangles[:, 1], triangles[:, 2])) / 6.0
        partition_volume = 0.0
        for index, target in enumerate(targets):
            selected = assignment == index
            if not np.any(selected):
                raise ValueError(f"No actual exterior faces assigned to {target.name}")
            target.mesh = arrays_to_mesh(world(vertices), faces[selected])
            target.surface_partition = True
            partition_volume += float(signed_volume[selected].sum())
            counts.append({"name": target.name, "faces": int(selected.sum())})
        if sum(item["faces"] for item in counts) != len(faces):
            raise ValueError(f"Exterior color partition lost source faces for {part.name}")
        return {"wire": part.name,
                "method": "mutually exclusive original exterior face subsets",
                "source_faces": len(faces), "display_faces": counts,
                "source_vertices_unchanged": True,
                "diagnostic_containment_tolerance_mm": tolerance_mm,
                "structural_stem_file": str(structural_stem_path),
                "structural_stem_sha256": sha256(structural_stem_path),
                "source_volume_mm3": float(signed_volume.sum()),
                "union_volume_difference_mm3": abs(partition_volume - float(signed_volume.sum()))}

    partition_report = []
    wires = metadata.get("mechanism", metadata)["release_wires"]
    for part in parts:
        if part.role != "wire":
            continue
        necks = [neck for neck in parts if neck.role == "neck"
                 and neck.name.startswith(part.name + "_neck_")]
        grips = [grip for grip in parts if grip.role == "grip"
                 and grip.name.startswith(part.name + "_grip_")]
        if not necks:
            continue
        original = solid(part.path)
        wire = wires[int(part.name.split("_")[1])]
        if grips and wire.get("structural_stem_mesh_file") and section_dimensions(wire, "neck"):
            partition_report.append(original_surface_partition(part, wire, necks, grips))
            continue
        regions = []
        color_parts = []
        neck_scales = []
        for neck in necks:
            endpoint = wire["endpoints"][int(neck.name.rsplit("_", 1)[1])]
            center = np.asarray(endpoint["neck_center"], dtype=float)
            axis = np.asarray(endpoint["inward_tangent"], dtype=float)
            axis /= np.linalg.norm(axis)
            scale = float(endpoint.get("display_neck_mask_radial_scale",
                                       wire.get("display_neck_mask_radial_scale",
                                                1.10)))
            radial_scale = np.eye(3) + (scale - 1.0) * (np.eye(3) - np.outer(axis, axis))
            transform = np.column_stack([radial_scale, center - radial_scale @ center])
            mask = solid(neck.path).transform(transform)
            regions.append(original ^ mask)
            color_parts.append(neck)
            neck_scales.append(scale)
        structural_stem_path = None
        structural_stem = None
        if wire.get("structural_stem_mesh_file"):
            structural_stem_path = Path(wire["structural_stem_mesh_file"])
            if not structural_stem_path.is_absolute():
                structural_stem_path = part.path.parent / structural_stem_path
            structural_stem = expanded_mask(solid(structural_stem_path), 1.0001)
        for grip in grips:
            region = original ^ expanded_mask(solid(grip.path), 1.0001)
            exclusions = list(regions)
            if structural_stem is not None:
                exclusions.append(structural_stem)
            if exclusions:
                region = mf.Manifold.batch_boolean([region] + exclusions, mf.OpType.Subtract)
            if region.is_empty():
                raise ValueError(f"The supplied grip has no visible contribution outside the structural stem: {grip.name}")
            regions.append(region)
            color_parts.append(grip)
        remainder = mf.Manifold.batch_boolean([original] + regions, mf.OpType.Subtract)
        union = mf.Manifold.batch_boolean([remainder] + regions, mf.OpType.Add)
        delta = abs(union.volume() - original.volume())
        if delta > 1e-7:
            raise ValueError(f"Display color partition changed {part.name} volume by {delta} mm3")
        part.mesh = display_mesh(remainder)
        for color_part, region in zip(color_parts, regions):
            color_part.mesh = display_mesh(region)
        partition_report.append({"wire": part.name,
                                 "source_volume_mm3": original.volume(),
                                 "diagnostic_neck_mask_radial_scales": neck_scales,
                                 "structural_stem_file": str(structural_stem_path) if structural_stem_path else None,
                                 "structural_stem_sha256": sha256(structural_stem_path) if structural_stem_path else None,
                                 "grip_and_stem_display_mask_scale": 1.0001 if grips else None,
                                 "union_volume_difference_mm3": delta})
    return partition_report


def partition_permanent_colors(parts, metadata, meta_path):
    """Color exact released layers as disjoint ivory and bearing regions."""
    import manifold3d as mf

    def resolve(entry):
        path = Path(file_name(entry))
        return path if path.is_absolute() else meta_path.parent / path

    def solid(path):
        with np.load(path, allow_pickle=False) as arrays:
            vertices = arrays["vertices"] if "vertices" in arrays.files else arrays["v"]
            faces = arrays["faces"] if "faces" in arrays.files else arrays["f"]
        return mf.Manifold(mf.Mesh64(np.asarray(vertices, dtype=np.float64),
                                     np.asarray(faces, dtype=np.uint32)))

    def display_mesh(model):
        source = model.to_mesh64()
        return arrays_to_mesh(world(source.vert_properties[:, :3]), source.tri_verts)

    data = metadata.get("mechanism", metadata)
    report, remove = [], []
    for layer, entry in enumerate(data["free_layer_meshes"]):
        source_path = resolve(entry)
        full = solid(source_path)
        shells = [part for part in parts if part.layer == layer and part.role == "shell"]
        features = [part for part in parts if part.layer == layer and part.role in ("bearing", "pivot")]
        if len(shells) != 1 or not features:
            raise ValueError(f"Missing actual shell / bearing display pieces for layer {layer}")
        feature_union = mf.Manifold.batch_boolean([solid(part.path) for part in features], mf.OpType.Add)
        colored = full ^ feature_union
        ivory = full - colored
        reunited = colored + ivory
        delta = abs(reunited.volume() - full.volume())
        if delta > 1e-6:
            raise ValueError(f"Layer {layer} display partition changed volume by {delta} mm3")
        shells[0].mesh = display_mesh(ivory)
        features[0].mesh = display_mesh(colored)
        features[0].role = "bearing"
        features[0].name = f"permanent_bearings_layer_{layer + 1:02d}"
        remove.extend(features[1:])
        report.append({"layer_index": layer, "source_layer_file": str(source_path),
                       "source_layer_sha256": sha256(source_path),
                       "feature_masks": [{"file": str(part.path), "sha256": sha256(part.path)} for part in features],
                       "source_volume_mm3": full.volume(),
                       "union_volume_difference_mm3": delta})
    parts[:] = [part for part in parts if not any(part is removed for removed in remove)]
    return report


def load_spec(meta_path):
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    spec = metadata.get("mechanism_render")
    if spec is None:
        spec = derive_spec(metadata, meta_path)
    parts = []
    for item in spec["parts"]:
        path = Path(item["file"])
        if not path.is_absolute():
            path = meta_path.parent / path
        role = item["role"]
        if role not in PALETTE:
            raise ValueError(f"Unknown actual-part color role: {role}")
        parts.append(Part(item.get("name", path.stem), role, path,
                          load_mesh(path, local_sphere=True), item.get("layer"), item.get("lighting", "crease")))
    spec["display_color_partition"] = partition_wire_colors(parts, metadata)
    spec["permanent_color_partition"] = partition_permanent_colors(parts, metadata, meta_path)
    return metadata, spec, parts


def overview(studio, spec):
    width, height = 1480, 1100
    panel_w, panel_h = 700, 780
    canvas = Image.new("RGB", (width, height), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), f"{spec['revision']['label']}  /  {spec['layer_count']}-LAYER ROTATING SPHERE", font=font(26), fill=INK)
    entries = [("shell", "CARVED SHELLS"), ("bearing", "PERMANENT BEARINGS"),
               ("wire", "REMOVE AFTER PRINT"), ("neck", "BREAKAWAY NECKS")]
    if spec["has_grip_parts"]:
        entries.append(("grip", "SEPARATE GRIP"))
    legend(draw, (34, 70), entries, step=int(1420 / len(entries)), fontsize=16)
    outer_radius = float(spec["outer_radius_mm"])
    scale = outer_radius * 1.26
    for index, (key, label) in enumerate([
        ("front", "WINDOW ACCESS"), ("section", "DISPLAY SECTION")]):
        pose = spec["views"][key]
        studio.highlight_wire(None)
        studio.show_roles()
        studio.section(pose.get("clip_normal_local"),
                       pose.get("clip_origin_local", [0, 0, 0]),
                       tuple(pose.get("clip_roles", ["shell"])))
        picture = studio.render((panel_w, panel_h), show_ground=False,
                                focus=world(pose.get("focus_local", [0, 0, 0])),
                                parallel_scale=float(pose.get("parallel_scale_mm", scale)),
                                **view_angles(pose["direction_local"]))
        for callout in pose.get("callouts", []):
            leader(picture, studio.project(callout["point_local"]),
                   callout["text"], tuple(callout["anchor_px"]),
                   rgb(callout.get("role", "bearing")), callout.get("right", False))
        x = 30 + index * 720
        canvas.paste(picture, (x, 153))
        draw.text((x + 12, 122), label, font=font(20), fill=INK)
    fit_text(draw, (38, 962), spec.get("overview_caption", ""), 1400, 19)
    fit_text(draw, (38, 1033),
             f"{spec['front_support_count']} front bridges + {spec['starter_support_count']} bottom starters. "
             "The right-hand shells are cut only for display; print geometry remains whole.",
             1400, 17)
    return canvas


def close_view(studio, spec, key, heading):
    pose = spec["views"][key]
    studio.highlight_wire(None)
    width, height = 1480, 1120
    canvas = Image.new("RGB", (width, height), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 25), heading, font=font(26), fill=INK)
    legend(draw, (34, 74), [("bearing", "PERMANENT"), ("wire", "REMOVABLE WIRE"),
                           ("neck", "BREAKAWAY NECK")], step=360)
    studio.show_roles(pose.get("visible_roles"), pose.get("visible_layers"))
    studio.section(pose.get("clip_normal_local"),
                   pose.get("clip_origin_local", [0, 0, 0]),
                   tuple(pose.get("clip_roles", ["shell"])), cap=pose.get("cap_section", False))
    picture = studio.render((1060, 850), show_ground=False,
                            focus=world(pose["focus_local"]),
                            parallel_scale=float(pose["parallel_scale_mm"]),
                            **view_angles(pose["direction_local"]))
    for callout in pose.get("callouts", []):
        leader(picture, studio.project(callout["point_local"]),
               callout["text"], tuple(callout["anchor_px"]),
               rgb(callout.get("role", "neck")), callout.get("right", False))
    for marker in pose.get("dimension_markers", []):
        dimension_marker(picture, studio.project(marker["p0_local"]),
                         studio.project(marker["p1_local"]), marker["name"],
                         marker.get("label_offset_px", [16, -20]))
    canvas.paste(picture, (20, 140))
    draw.rounded_rectangle((1094, 161, 1450, 966), radius=10, fill=PANEL)
    y = 187
    for row in pose.get("dimensions", []):
        draw.text((1115, y), row["label"], font=font(17), fill=QUIET)
        y += 31
        y += fit_text(draw, (1115, y), row["value"], 306, size=22, fill=INK)
        y += 24
    fit_text(draw, (36, 1015), pose.get("caption", ""), 1400, 19)
    return canvas


def release_view(studio, spec, starter=False, closeup=False):
    """Show each real endpoint through its own verified tool-entry direction."""
    endpoints = spec["starter_endpoints" if starter else "release_endpoints"]
    width, height = 1480, 1120
    canvas = Image.new("RGB", (width, height), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    selected = spec["selected_starter" if starter else "selected_wire"]
    heading = ("FLATTENED BREAK NECKS" if closeup and section_dimensions(selected, "neck") else
               "BOTTOM SHELL SERVICE ACCESS" if starter else "TWO ACCESSIBLE BREAKAWAY ENDS")
    draw.text((32, 24), spec["revision"]["label"] + "  /  " + heading, font=font(26), fill=INK)
    entries = [("wire", "SELECTED SUPPORT"), ("neck", "SELECTED BREAK NECK")]
    if spec["has_grip_parts"]:
        entries.append(("grip", "SEPARATE GRIP"))
    else:
        entries.insert(0, ("bearing", "PERMANENT BEARINGS"))
    legend(draw, (34, 70), entries, step=450)
    scale = 3.35
    if closeup:
        section = section_dimensions(selected, "neck")
        diameter = find_numeric(selected, ["neck_diameter_mm"])
        transverse = max(section) if section else diameter
        if transverse is None:
            raise ValueError("Neck close-up requires actual cross-section dimensions")
        scale = max(float(transverse) * 1.8, 0.7)
    for index, endpoint in enumerate(endpoints[:2]):
        neck = np.asarray(endpoint["neck_center"], dtype=float)
        direction = np.asarray(endpoint["tool_entry"], dtype=float) - neck
        studio.show_roles()
        studio.section(None)
        studio.highlight_wire(spec["selected_starter_part_prefix" if starter else "selected_wire_part_prefix"])
        picture = studio.render((700, 770), show_ground=False, focus=world(neck),
                                parallel_scale=scale, **view_angles(direction))
        neck_label = "FLATTENED NECK" if section_dimensions(selected, "neck") else "NARROW NECK"
        leader(picture, studio.project(neck), neck_label, (20, 27), rgb("neck"))
        x = 30 + index * 720
        canvas.paste(picture, (x, 156))
        draw.text((x + 12, 122), "HOST SHELL END" if index == 0 else "MOVING SHELL END",
                  font=font(20), fill=INK)
    for index, row in enumerate(spec.get("starter_dimensions" if starter else "release_dimensions", [])[:5]):
        x = 40 + 286 * index
        draw.text((x, 953), row["label"], font=font(16), fill=QUIET)
        draw.text((x, 985), row["value"], font=font(24), fill=INK)
    caption = ("Both contacts are viewed through their supplied tool-entry directions with the surrounding shells present. "
               "The highlighted support is removable after its two amber break necks are severed; the ivory shell pads remain.")
    fit_text(draw, (38, 1049), caption,
             1400, 18)
    return canvas


def starter_pair_view(studio, spec):
    """Show intact pair access beside an explicitly isolated actual-part view."""
    group = spec["selected_starter_group"]
    if len(group) < 2:
        return release_view(studio, spec, starter=True)
    pose = spec["views"]["starter_pair"]
    prefixes = spec["selected_starter_group_prefixes"]
    selected = spec["selected_starter"]
    width, height = 1480, 1120
    canvas = Image.new("RGB", (width, height), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    pair_name = "TWIN" if len(group) == 2 else str(len(group))
    draw.text((32, 24), f"{spec['revision']['label']}  /  {pair_name} STRAIGHT STARTERS AND SEPARATE GRIPS",
              font=font(25), fill=INK)
    legend(draw, (34, 70), [("shell", "SHELL PADS"), ("wire", "STRAIGHT STEMS"),
                           ("neck", "BREAK NECKS"), ("grip", "SEPARATE GRIPS")], step=355, fontsize=17)
    for index, (direction_key, label) in enumerate([
        ("front_direction_local", "INTACT SHELL ACCESS"),
        ("oblique_direction_local", "ACTUAL SUPPORT PAIR / ISOLATED OBLIQUE")]):
        studio.section(None)
        studio.show_roles()
        studio.highlight_wire(prefixes)
        if index == 1:
            studio.show_prefixes(prefixes)
        picture = studio.render((700, 770), show_ground=False,
                                focus=world(pose["focus_local"]),
                                parallel_scale=pose["parallel_scale_mm"],
                                **view_angles(pose[direction_key]))
        if index == 0:
            for wire in group:
                endpoints = wire["endpoints"]
                for endpoint in endpoints:
                    x, y = studio.project(endpoint["neck_center"])
                    marker = ImageDraw.Draw(picture)
                    marker.ellipse((x - 4, y - 4, x + 4, y + 4), outline=rgb("neck"), width=2)
        else:
            endpoints = sorted(selected["endpoints"],
                               key=lambda item: np.linalg.norm(item["attachment_point"]), reverse=True)
            leader(picture, studio.project(endpoints[0]["neck_center"]),
                   "FLATTENED NECK", (20, 690), rgb("neck"))
            if selected.get("grip_point") is not None:
                leader(picture, studio.project(selected["grip_point"]),
                       "SEPARATE GRIP", (680, 27), rgb("grip"), right=True)
        x = 30 + index * 720
        canvas.paste(picture, (x, 156))
        draw.text((x + 12, 122), label, font=font(17), fill=INK)
    for index, row in enumerate(spec.get("starter_dimensions", [])[:5]):
        x = 40 + 286 * index
        draw.text((x, 953), row["label"], font=font(16), fill=QUIET)
        draw.text((x, 985), row["value"], font=font(23), fill=INK)
    fit_text(draw, (38, 1049),
             "Left: access through the intact shell openings. Right: the actual support pair in its original positions, "
             "isolated and viewed obliquely to show the flattened necks and grips outside the straight stems.",
             1400, 18)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="?", type=Path,
                        default=ROOT / "cache" / "spheres" / "sphere_5layers.json")
    parser.add_argument("--out", type=Path, default=ROOT / "previews" / "mechanism")
    parser.add_argument("--only", nargs="+", choices=["overview", "release", "neck", "starter", "bearing"],
                        default=["overview", "neck", "starter", "bearing"])
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--no-report", action="store_true",
                        help="Write only the requested JPEG images")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metadata, spec, parts = load_spec(args.metadata)
    studio = MechanismStudio(parts, ao=not args.no_ao)
    report = {"revision": spec["revision"]["id"], "source_metadata": str(args.metadata),
              "metadata_sha256": sha256(args.metadata),
              "vtk_version": vtk.vtkVersion.GetVTKVersion(),
              "render_window": studio.window.GetClassName(),
              "source_parts": [{"name": p.name, "role": p.role, "file": str(p.path),
                                "sha256": sha256(p.path)} for p in parts],
              "display_only": "Source meshes retain their coordinates. Straight starter colors partition original exterior faces, each displayed once. Curved wire/neck colors and shell/bearing colors use disjoint exact Boolean regions. Lighting normals do not alter the actual surface. Clipping planes are explanatory.",
              "display_color_partition": spec.get("display_color_partition", []),
              "permanent_color_partition": spec.get("permanent_color_partition", []),
              "view_specification": spec.get("views", {}),
              "release_dimensions": spec.get("release_dimensions", []),
              "starter_dimensions": spec.get("starter_dimensions", []),
              "support_counts": {key: spec[key] for key in ("front_support_count", "starter_support_count", "support_count", "neck_count")},
              "images": []}
    try:
        for key in args.only:
            if key == "overview":
                image = overview(studio, spec)
                name = "mechanism_overview.jpg"
            elif key == "release":
                image = (release_view(studio, spec) if "release_endpoints" in spec else
                         close_view(studio, spec, "release", spec["revision"]["label"] + "  /  ACCESSIBLE BREAKAWAY SUPPORT"))
                name = "release_detail.jpg"
            elif key == "starter":
                if "starter_endpoints" not in spec:
                    raise ValueError("Actual starter-support endpoint metadata is not present")
                image = starter_pair_view(studio, spec)
                name = "starter_access.jpg"
            elif key == "neck":
                if "starter_endpoints" not in spec:
                    raise ValueError("Actual starter neck metadata is required for the reinforcement detail")
                image = release_view(studio, spec, starter=True, closeup=True)
                name = "neck_detail.jpg"
            else:
                image = close_view(studio, spec, "bearing", spec["revision"]["label"] + "  /  OPPOSED PIVOT BEARINGS")
                name = "bearing_section.jpg"
            path = args.out / name
            atomic_image(image, path)
            report["images"].append({"name": name, "size_px": list(image.size),
                                      "bytes": path.stat().st_size, "sha256": sha256(path),
                                      "mode": image.mode, "format": "JPEG", "baseline_jpeg": True})
            print(f"Rendered {name} ({time.time() - start:.1f} s)", flush=True)
        report["elapsed_seconds"] = round(time.time() - start, 2)
        report.update({"ao_enabled": studio.ao_pass is not None,
                       "ao_fallback_reason": studio.ao_fallback_reason,
                       "ao_unsupported_optional_settings": studio.ao_unsupported_optional_settings})
        if not args.no_report:
            (args.out / "mechanism_render_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        studio.close()


if __name__ == "__main__":
    main()
