"""Stream and verify full-resolution STL, core 3MF, and mobile-compatible GLB.

The input NPZ contains float32 ``vertices`` in millimetres, Z up, and an
integer ``faces`` array.  No repair, decimation, coordinate welding, or
winding changes take place.  The GLB rotates to Y up and converts to metres;
its position quantization error is measured against every input triangle.

Dependencies: numpy and vtk.  All output files are self-contained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import zipfile

import numpy as np

try:
    import resource
except ImportError:  # Windows does not expose the Unix resource module.
    resource = None


CORE_NS = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
MODEL_REL = 'http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel'
CONTENT_NS = 'http://schemas.openxmlformats.org/package/2006/content-types'
STL_RECORD = np.dtype([('normal', '<f4', (3,)),
                       ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
BLOCK = 32768


def log(message):
    print(message, flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def load_source(path):
    with np.load(path, allow_pickle=False) as data:
        vertices = np.ascontiguousarray(data['vertices'])
        faces = np.ascontiguousarray(data['faces'])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('Expected a nonempty vertices array with shape (n,3)')
    if vertices.dtype != np.dtype('float32'):
        raise ValueError('Authoritative input vertices must already be float32')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Expected a nonempty faces array with shape (n,3)')
    if faces.dtype.kind not in 'ui':
        raise ValueError('Face indices must be integers')
    if not np.isfinite(vertices).all():
        raise ValueError('Source contains nonfinite positions')
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError('A source face references an out-of-range vertex')
    if len(faces) >= 2 ** 32:
        raise ValueError('Binary STL triangle count exceeds uint32')
    return vertices, faces


def triangle_sha256(vertices, faces):
    digest = hashlib.sha256()
    for start in range(0, len(faces), BLOCK):
        p = np.ascontiguousarray(vertices[faces[start:start + BLOCK]], dtype='<f4')
        digest.update(p.tobytes())
    return digest.hexdigest()


def write_stl(path, vertices, faces):
    header = b'Ivory pen holder V5 rotating layers; mm; removable supports included'
    with Path(path).open('wb') as stream:
        stream.write(header.ljust(80, b' '))
        stream.write(struct.pack('<I', len(faces)))
        for start in range(0, len(faces), BLOCK):
            points = vertices[faces[start:start + BLOCK]]
            p64 = points.astype(np.float64)
            normals = np.cross(p64[:, 1] - p64[:, 0], p64[:, 2] - p64[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths == 0):
                raise ValueError('Authoritative input has an exactly degenerate triangle')
            normals /= lengths[:, None]
            records = np.zeros(len(points), dtype=STL_RECORD)
            records['normal'] = normals
            records['vertices'] = points
            stream.write(records.tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def verify_stl(path, vertices, faces, triangle_hash):
    with Path(path).open('rb') as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise ValueError('STL header truncated')
    count = struct.unpack('<I', header[80:84])[0]
    if count != len(faces) or Path(path).stat().st_size != 84 + 50 * count:
        raise ValueError('STL count or byte length mismatch')
    digest = hashlib.sha256()
    data = np.memmap(path, mode='r', dtype=STL_RECORD, offset=84, shape=(count,))
    for start in range(0, count, BLOCK):
        records = data[start:start + BLOCK]
        actual = records['vertices']
        expected = vertices[faces[start:start + BLOCK]]
        if not np.array_equal(actual, expected):
            raise ValueError(f'STL triangle coordinate/winding mismatch at {start}')
        if not np.isfinite(records['normal']).all() or np.any(records['attribute']):
            raise ValueError('STL has invalid normals or nonzero attribute bytes')
        digest.update(np.ascontiguousarray(actual, dtype='<f4').tobytes())
    del data
    if digest.hexdigest() != triangle_hash:
        raise ValueError('STL canonical triangle digest mismatch')
    return {'triangle_count': count, 'all_triangle_coordinates_exact': True,
            'triangle_order_and_winding_preserved': True,
            'canonical_triangle_sha256': digest.hexdigest(), 'units': 'mm'}


def write_3mf(path, vertices, faces, title):
    content_types = (f'<?xml version="1.0" encoding="UTF-8"?>'
                     f'<Types xmlns="{CONTENT_NS}">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                     '</Types>')
    relationships = (f'<?xml version="1.0" encoding="UTF-8"?>'
                     f'<Relationships xmlns="{REL_NS}">'
                     f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{MODEL_REL}"/>'
                     '</Relationships>')
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        archive.writestr('[Content_Types].xml', content_types)
        archive.writestr('_rels/.rels', relationships)
        with archive.open('3D/3dmodel.model', 'w', force_zip64=True) as stream:
            prefix = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                      f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}">\n'
                      f'<metadata name="Title">{escape(title)}</metadata>\n'
                      '<metadata name="Application">Validated full-resolution mesh export</metadata>\n'
                      '<resources><object id="1" type="model"><mesh><vertices>\n')
            stream.write(prefix.encode('utf-8'))
            # Seventeen digits express each float32 coordinate exactly when
            # parsed as float64; this also avoids a merely approximate XML
            # position match hidden by a later float32 cast.
            for start in range(0, len(vertices), 4096):
                block = ''.join(f'<vertex x="{float(x):.17g}" y="{float(y):.17g}" z="{float(z):.17g}"/>\n'
                                for x, y, z in vertices[start:start + 4096])
                stream.write(block.encode('ascii'))
            stream.write(b'</vertices><triangles>\n')
            for start in range(0, len(faces), 8192):
                block = ''.join(f'<triangle v1="{int(a)}" v2="{int(b)}" v3="{int(c)}"/>\n'
                                for a, b, c in faces[start:start + 8192])
                stream.write(block.encode('ascii'))
            stream.write(b'</triangles></mesh></object></resources>'
                         b'<build><item objectid="1"/></build></model>\n')


def verify_3mf(path, vertices, faces, triangle_hash):
    vertex_count = face_count = 0
    vertex_buffer, face_buffer = [], []
    found_model = False
    object_ids, build_ids = [], []

    def compare_vertices():
        nonlocal vertex_count
        count = len(vertex_buffer)
        actual = np.asarray(vertex_buffer, dtype=np.float64)
        expected = vertices[vertex_count:vertex_count + count].astype(np.float64)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f'3MF exact vertex coordinate mismatch at {vertex_count}')
        vertex_count += count
        vertex_buffer.clear()

    def compare_faces():
        nonlocal face_count
        count = len(face_buffer)
        actual = np.asarray(face_buffer, dtype=np.uint64)
        expected = faces[face_count:face_count + count]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f'3MF face index/winding mismatch at {face_count}')
        face_count += count
        face_buffer.clear()

    with zipfile.ZipFile(path, 'r') as archive:
        if set(archive.namelist()) != {'[Content_Types].xml', '_rels/.rels', '3D/3dmodel.model'}:
            raise ValueError('Unexpected or missing core3MF ZIP members')
        content = ET.fromstring(archive.read('[Content_Types].xml'))
        rels = ET.fromstring(archive.read('_rels/.rels'))
        if content.tag != f'{{{CONTENT_NS}}}Types':
            raise ValueError('3MF content-types namespace mismatch')
        relationship = rels.find(f'{{{REL_NS}}}Relationship')
        if relationship is None or relationship.get('Type') != MODEL_REL or relationship.get('Target') != '/3D/3dmodel.model':
            raise ValueError('3MF model relationship missing or incorrect')
        with archive.open('3D/3dmodel.model') as stream:
            stack = []
            for event, element in ET.iterparse(stream, events=('start', 'end')):
                if event == 'start':
                    stack.append(element)
                    if not found_model:
                        if element.tag != f'{{{CORE_NS}}}model' or element.get('unit') != 'millimeter':
                            raise ValueError('3MF model namespace/units mismatch')
                        found_model = True
                    continue
                tag = element.tag.rsplit('}', 1)[-1]
                if tag == 'vertex':
                    vertex_buffer.append(tuple(float(element.attrib[k]) for k in ('x', 'y', 'z')))
                    if len(vertex_buffer) == BLOCK:
                        compare_vertices()
                elif tag == 'triangle':
                    face_buffer.append(tuple(int(element.attrib[k]) for k in ('v1', 'v2', 'v3')))
                    if len(face_buffer) == BLOCK:
                        compare_faces()
                elif tag == 'object':
                    object_ids.append(element.get('id'))
                elif tag == 'item':
                    build_ids.append(element.get('objectid'))
                    if 'transform' in element.attrib:
                        raise ValueError('Unexpected 3MF build transform')
                # Removing completed children keeps ElementTree memory bounded;
                # clearing only their contents would retain millions of nodes.
                if len(stack) > 1:
                    stack[-2].remove(element)
                element.clear()
                stack.pop()
            if vertex_buffer:
                compare_vertices()
            if face_buffer:
                compare_faces()
    if vertex_count != len(vertices) or face_count != len(faces):
        raise ValueError('3MF total vertex/face count mismatch')
    if object_ids != ['1'] or build_ids != ['1']:
        raise ValueError('3MF must contain exactly one model object and build item')
    return {'vertex_count': vertex_count, 'triangle_count': face_count,
            'all_vertex_coordinates_exact_as_float64': True,
            'triangle_indices_order_and_winding_preserved': True,
            'canonical_triangle_sha256': triangle_hash,
            'units': 'millimeter', 'core_only': True,
            'xml_validation': 'streamed every vertex and triangle'}


def crease_normals(vertices, faces, angle=35.0):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    index_dtype = np.int32 if max(len(vertices), faces.size) < 2 ** 31 else np.int64
    connectivity = np.ascontiguousarray(faces.reshape(-1), dtype=index_dtype)
    offsets = np.arange(0, connectivity.size + 1, 3, dtype=index_dtype)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=False))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=False), numpy_to_vtk(connectivity, deep=False))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    pipeline = vtk.vtkPolyDataNormals()
    pipeline.SetInputData(poly)
    pipeline.SetFeatureAngle(angle)
    pipeline.SplittingOn()
    # The source already has authoritative outward winding.  Consistency and
    # automatic reorientation could silently change it, so both are disabled.
    pipeline.ConsistencyOff()
    pipeline.AutoOrientNormalsOff()
    pipeline.FlipNormalsOff()
    pipeline.NonManifoldTraversalOff()
    pipeline.ComputePointNormalsOn()
    pipeline.ComputeCellNormalsOff()
    pipeline.Update()
    output = pipeline.GetOutput()
    split_vertices = np.array(vtk_to_numpy(output.GetPoints().GetData()), dtype=np.float32, copy=True)
    normals = np.array(vtk_to_numpy(output.GetPointData().GetNormals()), dtype=np.float32, copy=True)
    connectivity_out = vtk_to_numpy(output.GetPolys().GetConnectivityArray())
    offsets_out = vtk_to_numpy(output.GetPolys().GetOffsetsArray())
    if len(offsets_out) != len(faces) + 1 or not np.all(np.diff(offsets_out) == 3):
        raise ValueError('Normal splitting unexpectedly changed triangle topology')
    split_faces = np.array(connectivity_out, copy=True).reshape((-1, 3))
    for start in range(0, len(faces), BLOCK):
        if not np.array_equal(split_vertices[split_faces[start:start + BLOCK]],
                              vertices[faces[start:start + BLOCK]]):
            raise ValueError('Normal splitting changed triangle coordinates, order, or winding')
    if not np.isfinite(normals).all():
        raise ValueError('Normal generation produced nonfinite values')
    length = np.linalg.norm(normals.astype(np.float64), axis=1)
    used = np.zeros(len(normals), dtype=bool)
    used[split_faces.reshape(-1)] = True
    bad = (length < 1e-12) & used
    fallback_count = int(bad.sum())
    if fallback_count:
        # VTK may underflow the normal of extremely small Boolean triangles.
        # Recompute only affected display normals from the unchanged split
        # topology using float64 cross products and incident-face averaging.
        lookup = np.full(len(normals), -1, dtype=np.int32)
        lookup[bad] = np.arange(fallback_count, dtype=np.int32)
        sums = np.zeros((fallback_count, 3), dtype=np.float64)
        largest_area = np.zeros(fallback_count, dtype=np.float64)
        largest_normal = np.zeros_like(sums)
        for start in range(0, len(split_faces), BLOCK):
            face = split_faces[start:start + BLOCK]
            local = lookup[face]
            selected = np.any(local >= 0, axis=1)
            if not np.any(selected):
                continue
            local = local[selected]
            p = split_vertices[face[selected]].astype(np.float64)
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            area = np.linalg.norm(n, axis=1)
            nonzero = area > 0
            n[nonzero] /= area[nonzero, None]
            for corner in range(3):
                keep = (local[:, corner] >= 0) & nonzero
                ids = local[keep, corner]
                if not len(ids):
                    continue
                np.add.at(sums, ids, n[keep])
                np.maximum.at(largest_area, ids, area[keep])
                best = area[keep] >= largest_area[ids]
                largest_normal[ids[best]] = n[keep][best]
        magnitude = np.linalg.norm(sums, axis=1)
        cancelled = magnitude < 1e-12
        sums[cancelled] = largest_normal[cancelled]
        magnitude = np.linalg.norm(sums, axis=1)
        if np.any(magnitude < 1e-12):
            raise ValueError('A used point has no incident nondegenerate face normal')
        normals[bad] = sums / magnitude[:, None]
        length[bad] = np.linalg.norm(normals[bad].astype(np.float64), axis=1)
    # VTK retains source points not referenced by any triangle.  They never
    # enter a GLB primitive, but give them a finite harmless normal before the
    # vectorized normalization step.
    normals[~used] = [0, 0, 1]
    length[~used] = 1
    # Explicitly normalize after VTK's float32 summation; this changes only
    # display normals, never positions or face winding.
    normals = (normals.astype(np.float64) / length[:, None]).astype(np.float32)
    return split_vertices, split_faces, normals, fallback_count


def _write_binary_array(stream, array, views, target):
    offset = stream.tell()
    if offset % 4:
        stream.write(b'\0' * (-offset % 4))
        offset = stream.tell()
    payload = array.tobytes(order='C')
    stream.write(payload)
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(payload), 'target': target})
    return len(views) - 1


def write_glb(path, vertices, faces, title):
    split_vertices, split_faces, normals, fallback_count = crease_normals(vertices, faces)
    binary_path = Path(str(path) + '.bin.tmp')
    views, accessors, primitives = [], [], []
    maximum_vertices = 0
    try:
        with binary_path.open('wb') as binary:
            # <=20,000 triangles guarantees <=60,000 distinct points even
            # when every corner is split at a crease.  No uint32 indices or
            # optional glTF extensions are necessary.
            for start in range(0, len(faces), 20000):
                triangle_indices = split_faces[start:start + 20000]
                ids, inverse = np.unique(triangle_indices.reshape(-1), return_inverse=True)
                maximum_vertices = max(maximum_vertices, len(ids))
                position = split_vertices[ids].astype(np.float64)
                position = position[:, [0, 2, 1]]
                position[:, 2] *= -1
                position = np.ascontiguousarray(position * 0.001, dtype='<f4')
                normal = normals[ids][:, [0, 2, 1]].copy()
                normal[:, 2] *= -1
                normal = np.ascontiguousarray(normal, dtype='<f4')
                index = np.ascontiguousarray(inverse, dtype='<u2')
                position_view = _write_binary_array(binary, position, views, 34962)
                normal_view = _write_binary_array(binary, normal, views, 34962)
                index_view = _write_binary_array(binary, index, views, 34963)
                p_accessor = len(accessors)
                accessors.append({'bufferView': position_view, 'componentType': 5126,
                                  'count': len(ids), 'type': 'VEC3',
                                  'min': position.min(axis=0).astype(float).tolist(),
                                  'max': position.max(axis=0).astype(float).tolist()})
                n_accessor = len(accessors)
                accessors.append({'bufferView': normal_view, 'componentType': 5126,
                                  'count': len(ids), 'type': 'VEC3'})
                i_accessor = len(accessors)
                accessors.append({'bufferView': index_view, 'componentType': 5123,
                                  'count': len(index), 'type': 'SCALAR',
                                  'min': [int(index.min())], 'max': [int(index.max())]})
                primitives.append({'attributes': {'POSITION': p_accessor, 'NORMAL': n_accessor},
                                   'indices': i_accessor, 'material': 0, 'mode': 4})
            if binary.tell() % 4:
                binary.write(b'\0' * (-binary.tell() % 4))
        binary_length = binary_path.stat().st_size
        document = {
            'asset': {'version': '2.0', 'generator': 'Validated full-resolution mesh export'},
            'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0, 'name': title}],
            'meshes': [{'name': title, 'primitives': primitives}],
            'materials': [{'name': 'Warm ivory', 'alphaMode': 'OPAQUE', 'doubleSided': False,
                           'pbrMetallicRoughness': {'baseColorFactor': [0.84, 0.80, 0.69, 1.0],
                                                   'metallicFactor': 0.0, 'roughnessFactor': 0.58}}],
            'buffers': [{'byteLength': binary_length}],
            'bufferViews': views, 'accessors': accessors,
        }
        encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        encoded += b' ' * (-len(encoded) % 4)
        total_length = 12 + 8 + len(encoded) + 8 + binary_length
        if total_length >= 2 ** 32:
            raise ValueError('GLB total length exceeds uint32')
        with Path(path).open('wb') as destination:
            destination.write(struct.pack('<4sII', b'glTF', 2, total_length))
            destination.write(struct.pack('<II', len(encoded), 0x4E4F534A))
            destination.write(encoded)
            destination.write(struct.pack('<II', binary_length, 0x004E4942))
            with binary_path.open('rb') as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        return {'normal_feature_angle_degrees': 35.0,
                'float64_recomputed_display_normals': fallback_count,
                'normal_split_vertex_count': len(split_vertices),
                'primitive_count': len(primitives),
                'maximum_primitive_vertices': maximum_vertices}
    finally:
        binary_path.unlink(missing_ok=True)


def verify_glb(path, vertices, faces, tolerance_mm=1e-5):
    total_faces = 0
    max_error = 0.0
    max_vertices = 0
    with Path(path).open('rb') as stream:
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, total = struct.unpack_from('<4sII', data, 0)
        if magic != b'glTF' or version != 2 or total != len(data):
            raise ValueError('GLB header or byte length mismatch')
        json_length, json_type = struct.unpack_from('<II', data, 12)
        if json_type != 0x4E4F534A or json_length % 4:
            raise ValueError('GLB JSON chunk invalid')
        document = json.loads(data[20:20 + json_length].decode('utf-8'))
        bin_header = 20 + json_length
        bin_length, bin_type = struct.unpack_from('<II', data, bin_header)
        binary_offset = bin_header + 8
        if bin_type != 0x004E4942 or binary_offset + bin_length != total or bin_length % 4:
            raise ValueError('GLB binary chunk invalid')
        if len(document['buffers']) != 1 or document['buffers'][0].get('uri') is not None:
            raise ValueError('GLB contains an external or additional buffer')
        if document['buffers'][0]['byteLength'] != bin_length:
            raise ValueError('GLB declared buffer length mismatch')
        if document.get('extensionsUsed') or document.get('extensionsRequired'):
            raise ValueError('Unexpected GLB extension dependency')
        if document.get('images') or document.get('textures'):
            raise ValueError('Unexpected GLB texture dependency')
        if len(document['meshes']) != 1 or len(document['nodes']) != 1:
            raise ValueError('GLB must contain one mesh node')
        if any(key in document['nodes'][0] for key in ('matrix', 'translation', 'rotation', 'scale')):
            raise ValueError('Unexpected GLB node transform')

        def accessor_array(index, expected_type, component_type):
            accessor = document['accessors'][index]
            view = document['bufferViews'][accessor['bufferView']]
            if accessor['type'] != expected_type or accessor['componentType'] != component_type:
                raise ValueError('GLB accessor type mismatch')
            if 'sparse' in accessor or 'byteStride' in view or view['buffer'] != 0:
                raise ValueError('GLB accessor unexpectedly sparse/interleaved/external')
            dtype = np.dtype('<f4' if component_type == 5126 else '<u2')
            width = 3 if expected_type == 'VEC3' else 1
            length = accessor['count'] * width * dtype.itemsize
            offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
            if offset % dtype.itemsize or length > view['byteLength'] or offset + length > bin_length:
                raise ValueError('GLB accessor buffer range mismatch')
            shape = (accessor['count'], 3) if width == 3 else (accessor['count'],)
            return np.ndarray(shape=shape, dtype=dtype, buffer=data, offset=binary_offset + offset)

        for primitive in document['meshes'][0]['primitives']:
            if primitive.get('mode', 4) != 4:
                raise ValueError('GLB primitive is not triangles')
            attributes = primitive['attributes']
            position = accessor_array(attributes['POSITION'], 'VEC3', 5126)
            normal = accessor_array(attributes['NORMAL'], 'VEC3', 5126)
            index = accessor_array(primitive['indices'], 'SCALAR', 5123)
            if len(position) > 65535 or len(position) != len(normal) or len(index) % 3:
                raise ValueError('GLB primitive size/index layout mismatch')
            if int(index.max()) >= len(position):
                raise ValueError('GLB index out of range')
            if not np.isfinite(position).all() or not np.isfinite(normal).all():
                raise ValueError('GLB contains nonfinite attributes')
            if not np.allclose(np.linalg.norm(normal.astype(np.float64), axis=1), 1, rtol=0, atol=2e-6):
                raise ValueError('GLB normals are not explicitly normalized')
            count = len(index) // 3
            actual = position[index.reshape((-1, 3))].astype(np.float64)
            # Inverse of [x,y,z] mm -> [x,z,-y]/1000 metres.
            actual = actual[:, :, [0, 2, 1]]
            actual[:, :, 1] *= -1
            actual *= 1000.0
            expected = vertices[faces[total_faces:total_faces + count]].astype(np.float64)
            if actual.shape != expected.shape:
                raise ValueError('GLB contains extra triangles')
            error = float(np.max(np.abs(actual - expected)))
            max_error = max(max_error, error)
            if error > tolerance_mm:
                raise ValueError(f'GLB triangle coordinate/winding mismatch: {error:g} mm')
            total_faces += count
            max_vertices = max(max_vertices, len(position))
            del position, normal, index
        if total_faces != len(faces):
            raise ValueError('GLB triangle count mismatch')
        data.close()
    return {'triangle_count': total_faces,
            'all_triangle_coordinates_verified': True,
            'triangle_order_and_winding_preserved': True,
            'maximum_coordinate_error_mm': max_error,
            'coordinate_error_tolerance_mm': tolerance_mm,
            'maximum_primitive_vertices': max_vertices,
            'index_component_type': 5123, 'index_type': 'uint16',
            'units': 'metre', 'up_axis': '+Y',
            'source_mm_to_glb_m': '[x, z, -y] / 1000',
            'external_resources': False, 'required_extensions': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stem', default='ivory_multisphere_pen_holder_v5_printpath_20261025')
    args = parser.parse_args()
    if not args.stem or Path(args.stem).name != args.stem:
        raise ValueError('--stem must be a filename without directories')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    vertices, faces = load_source(args.input)
    canonical_hash = triangle_sha256(vertices, faces)
    log(f'Loaded {len(vertices):,} vertices and {len(faces):,} triangles; source is unchanged')
    manifest = {
        'source': args.input.name, 'source_sha256': file_sha256(args.input),
        'source_vertex_count': len(vertices), 'source_triangle_count': len(faces),
        'source_coordinates': {'units': 'millimetre', 'up_axis': '+Z', 'dtype': 'float32'},
        'source_bounds_mm': [vertices.min(axis=0).astype(float).tolist(), vertices.max(axis=0).astype(float).tolist()],
        'canonical_triangle_sha256': canonical_hash, 'decimation': False, 'files': {},
    }
    title = 'Ivory multisphere pen holder V5 with rotating layers and breakaway supports'
    for extension in ('stl', '3mf', 'glb'):
        target = args.out / f'{args.stem}.{extension}'
        temporary = Path(str(target) + '.part')
        log(f'Writing {extension.upper()}...')
        if extension == 'stl':
            write_stl(temporary, vertices, faces)
            log('Verifying every STL triangle...')
            validation = verify_stl(temporary, vertices, faces, canonical_hash)
        elif extension == '3mf':
            write_3mf(temporary, vertices, faces, title)
            log('Verifying streamed 3MF vertices, indices, package and units...')
            validation = verify_3mf(temporary, vertices, faces, canonical_hash)
        else:
            details = write_glb(temporary, vertices, faces, title)
            log('Verifying all GLB primitives, normals and transformed triangles...')
            validation = verify_glb(temporary, vertices, faces)
            validation.update(details)
        temporary.replace(target)
        validation['bytes'] = target.stat().st_size
        validation['sha256'] = file_sha256(target)
        manifest['files'][target.name] = validation
        log(f'Validated {target.name}: {target.stat().st_size / 1e6:.2f} MB')
    manifest['elapsed_seconds'] = time.monotonic() - started
    if resource is not None:
        manifest['peak_resident_memory_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    manifest_path = args.out / f'{args.stem}_export_validation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log(f'Finished and validated all exports in {manifest["elapsed_seconds"]:.1f} seconds')
    log(str(manifest_path.resolve()))


if __name__ == '__main__':
    main()
