#!/usr/bin/env python3
"""Small geometry-only equivalence checks for the lazy B3 adapter."""
from pathlib import Path
import hashlib
import json
import time

import manifold3d as mf
import numpy as np

import custom_holder_supports as H
import external_removal_prism_stream_v6 as S

ROOT = Path(__file__).resolve().parent.parent
B3, B = S.B3, S.B


def check(name, solid, direction, distance):
    started = time.monotonic()
    early = S.iter_translation_sweep(solid, direction, distance)
    before_rejected = False
    try:
        early.coverage_proof()
    except S.CertificateError:
        before_rejected = True
    first, initial_record = next(early)
    after_initial = early.partial_status()
    initial_only_rejected = False
    try:
        early.coverage_proof()
    except S.CertificateError:
        initial_only_rejected = True
    assert first is solid and initial_record['component_index'] == 0
    assert before_rejected and initial_only_rejected
    assert after_initial['constructed_prism_count'] == 0 and not after_initial['exhausted']
    batch, batch_proof = B3.translation_sweep_components(solid, direction, distance)
    stream = S.iter_translation_sweep(solid, direction, distance)
    streamed = []
    component_records = []
    for part, record in stream:
        streamed.append(part)
        component_records.append(record)
    proof = stream.coverage_proof()
    assert stream.exhausted and len(streamed) == len(batch) == proof['verification_component_count']
    assert proof == batch_proof
    oriented_hashes = []
    raw_array_identity = []
    for index, (a, b, record) in enumerate(zip(batch, streamed, component_records)):
        assert record['component_index'] == index
        a_hash = B._oriented_triangles_sha(a)
        b_hash = B._oriented_triangles_sha(b)
        assert a_hash == b_hash
        av, af = B._arrays(a)
        bv, bf = B._arrays(b)
        raw_array_identity.append(bool(np.array_equal(av, bv) and np.array_equal(af, bf)))
        oriented_hashes.append(a_hash)
    one_prism_status = None
    if len(batch) > 1:
        next(early)
        one_prism_status = early.partial_status()
        assert one_prism_status['constructed_prism_count'] == 1
        assert one_prism_status['yielded_component_count'] == 2
        assert not one_prism_status['exhausted']
        try:
            early.coverage_proof()
        except S.CertificateError:
            pass
        else:
            raise AssertionError('Incomplete stream exposed a full proof')
    return dict(name=name, passed=True, component_count=len(batch),
        identical_oriented_triangle_component_count=len(oriented_hashes),
        identical_raw_vertex_face_array_component_count=sum(raw_array_identity),
        ordered_component_geometry_digest=hashlib.sha256(json.dumps(oriented_hashes).encode()).hexdigest(),
        exact_batch_summary_equality=True, batch_summary_sha256=hashlib.sha256(
            json.dumps(batch_proof, sort_keys=True).encode()).hexdigest(),
        prism_derivation_records_sha256=proof['prism_derivation_records_sha256'],
        exactly_zero_face_count=proof['exactly_zero_volume_face_count'],
        initial_signed_solid_object_preserved=True,
        initial_oriented_component_signed_volumes_mm3=[float(p.volume()) for p in solid.decompose()],
        initial_only_status=after_initial, first_prism_only_status=one_prism_status,
        incomplete_proof_rejected=True, seconds=time.monotonic() - started)


def main():
    fixture_path = ROOT / 'reports/external_removal_collision_attribution_cac74327_v6.json'
    fixture = json.loads(fixture_path.read_text())
    leg = next(row for row in fixture['results'] if row['id'] == 'S4_A040_051')['legs'][0]
    partition_path = ROOT / 'reports/sphere_removal_partition_v6.json'
    partition = json.loads(partition_path.read_text())
    row = next(row for row in partition['records']
               if row['layer_count'] == 4 and row['support_index'] == 51 and row['placement'] == 3)
    entry = row['paths']['removable']
    mesh_path = ROOT / entry['path']
    assert hashlib.sha256(mesh_path.read_bytes()).hexdigest() == entry['sha256']
    actual = H.load(mesh_path)[0]
    assert B._oriented_triangles_sha(actual) == leg['sweep_construction']['starting_oriented_triangles_sha256']
    cavity = mf.Manifold.cube((1., 1., 1.)) - mf.Manifold.cube((.4, .4, .4)).translate((.3, .3, .3))
    results = [check('actual_S4_A040_051', actual, leg['direction'], leg['distance_mm']),
               check('signed_cavity_with_parallel_faces', cavity, [1., 0., 0.], .5),
               check('zero_translation_signed_cavity', cavity, [1., 0., 0.], 0.)]
    report = dict(schema='streaming_prism_batch_equivalence_v1', passed=True,
        scope='Three small geometry-only cases; no scene collision or route acceptance checks.',
        source_bindings=S.source_bindings(), validation_source=dict(
            path=str(Path(__file__).relative_to(ROOT)),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        input_mesh=entry, input_fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        input_partition_sha256=hashlib.sha256(partition_path.read_bytes()).hexdigest(),
        records=results, frozen_B_and_B3_modified=False, active_sources_modified=False,
        original_or_full_scene_loaded=False)
    output = ROOT / 'reports/external_removal_prism_stream_equivalence_v6.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(output.relative_to(ROOT)),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), source=S.source_bindings()[0],
        results=[{k: row[k] for k in ['name', 'component_count', 'identical_raw_vertex_face_array_component_count',
                                     'exact_batch_summary_equality', 'exactly_zero_face_count', 'seconds']}
                 for row in results]), indent=2), flush=True)


if __name__ == '__main__':
    main()
