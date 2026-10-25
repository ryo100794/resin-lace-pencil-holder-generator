#!/usr/bin/env python3
"""Lazy, order-identical evaluation of frozen B3 translation components.

The iterator yields (solid, component_record). Component zero is the literal
signed starting solid. Later components use B3.checked_prism unchanged, in
the same nonzero boundary-face order. A consumer may reject early; it must
not accept until exhaustion, coverage_proof(), and every complete-scene
component overlap (including duplicates) satisfy the unchanged SUM gate.

coverage_proof() is unavailable after early stopping or a constructor error.
After normal exhaustion it is identical to B3's batch coverage summary. The
separate source_bindings() ledger includes this adapter, B3 and original B.
No existing source, actual stock, scene state or cut definition is modified.
"""
from pathlib import Path
from fractions import Fraction
import copy
import hashlib
import json
import math

import numpy as np

import external_removal_prisms_v6_b3 as B3

ROOT = Path(__file__).resolve().parent.parent
B = B3.B
CertificateError = B3.CertificateError
EXPECTED_B3_SHA256 = '80382dd83c97a8802756c47cf332ffb8cbefb049f1bc9d08945dd5dc7358f6c3'


def source_bindings():
    """Return adapter+B3+B hashes without constructing any geometry."""
    base = B3.source_bindings()
    if base[0]['sha256'] != EXPECTED_B3_SHA256:
        raise CertificateError('Streaming adapter requires the reviewed immutable B3 source')
    own = Path(__file__).resolve()
    return [dict(path=str(own.relative_to(ROOT)),
                 sha256=hashlib.sha256(own.read_bytes()).hexdigest())] + base


class TranslationSweepIterator:
    def __init__(self, solid, direction, distance):
        self._source_bindings = source_bindings()
        self._batch_bindings = copy.deepcopy(self._source_bindings[1:])
        B._valid(solid, 'Starting actual sweep stock')
        direction = np.asarray(direction, float)
        distance = float(distance)
        if (direction.shape != (3,) or not np.all(np.isfinite(direction))
                or not math.isfinite(distance) or distance < 0.
                or abs(float(np.linalg.norm(direction)) - 1.) > 1e-8):
            raise CertificateError('Translation requires a finite unit direction and nonnegative distance')
        self._solid = solid
        self._direction = direction
        self._distance = distance
        self._delta = direction * distance
        self._vertices, self._faces = B._arrays(solid)
        _, _, self._initial_serialization = B._roundtrip(solid)
        self._qvertices = [tuple(Fraction.from_float(float(x)) for x in vertex)
                           for vertex in self._vertices]
        self._qdelta = tuple(Fraction.from_float(float(x)) for x in self._delta)
        self._derivations = []
        self._zero_faces = []
        self._face_cursor = 0
        self._yielded_components = 0
        self._initial_yielded = False
        self._finished = False
        self._failure = None
        self._proof = None

    @property
    def source_bindings(self):
        return copy.deepcopy(self._source_bindings)

    @property
    def exhausted(self):
        return self._finished

    def __iter__(self):
        return self

    def __next__(self):
        if self._failure is not None:
            raise CertificateError('A prior constructor failure permanently invalidated this stream',
                                   dict(prior_error=self._failure))
        if self._finished:
            raise StopIteration
        if not self._initial_yielded:
            self._initial_yielded = True
            self._yielded_components = 1
            return self._solid, dict(component_index=0, actual_signed_starting_stock=True,
                                     query_is_original_convex_constructor_object=False)
        try:
            while self._face_cursor < len(self._faces):
                face_index = self._face_cursor
                face = self._faces[face_index]
                self._face_cursor += 1
                qa, qb, qc = [self._qvertices[int(i)] for i in face]
                determinant = B._exact_det(qa, qb, qc, self._qdelta)
                if determinant == 0:
                    self._zero_faces.append(int(face_index))
                    continue
                expected = float(abs(determinant) / 2)
                if expected <= 0. or not math.isfinite(expected):
                    raise CertificateError('Nonzero exact prism volume is not representable',
                                           dict(face_index=int(face_index)))
                triangle = self._vertices[face]
                ideals = [qa, qb, qc] + [tuple(point[k] + self._qdelta[k] for k in range(3))
                                        for point in (qa, qb, qc)]
                prism, certificate = B3.checked_prism(np.vstack([triangle, triangle + self._delta]),
                                                    ideals, self._delta, face_index)
                rt = certificate['serialization']
                # This record is deliberately field-for-field identical to
                # frozen B3's batch derivation. The caller receives a copy.
                record = dict(face_index=int(face_index), component_index=self._yielded_components,
                    expected_prism_volume_mm3=expected, constructed_bound_volume_mm3=B._volume(prism),
                    conservative_extra_volume_mm3=max(0., float(prism.volume()) - expected),
                    expected_volume_error_mm3=abs(float(prism.volume()) - expected),
                    serialized_sha256=rt['serialized_sha256'],
                    identical_oriented_triangles=rt['identical_oriented_triangles'],
                    serialization_uncovered_mm3=rt['uncovered_source_volume_mm3'],
                    **{key: value for key, value in certificate.items() if key != 'serialization'})
                self._derivations.append(record)
                self._yielded_components += 1
                return prism, copy.deepcopy(record)
            # Source drift must not turn a partially constructed stream into
            # a certificate for a different implementation.
            if source_bindings() != self._source_bindings:
                raise CertificateError('Bound streaming sources changed during iteration')
            self._proof = self._make_batch_equivalent_proof()
            self._finished = True
        except Exception as error:
            # In particular, retrying next() after an error cannot skip the
            # failed face and then manufacture a successful coverage proof.
            self._failure = f'{type(error).__name__}: {error}'
            raise
        raise StopIteration

    def partial_status(self):
        """Diagnostic progress only. An incomplete status is never a proof."""
        return dict(exhausted=self._finished, failed=self._failure is not None,
            failure=self._failure, yielded_component_count=self._yielded_components,
            constructed_prism_count=len(self._derivations),
            processed_boundary_face_count=self._face_cursor,
            total_boundary_face_count=len(self._faces),
            exactly_zero_volume_face_count=len(self._zero_faces),
            full_coverage_proof_available=self._finished and self._failure is None)

    def coverage_proof(self):
        if not self._finished or self._failure is not None or self._proof is None:
            raise CertificateError('No complete coverage proof before successful iterator exhaustion',
                                   self.partial_status())
        return copy.deepcopy(self._proof)

    def _make_batch_equivalent_proof(self):
        """Frozen B3 summary, without rebuilding any already-yielded prism."""
        bindings = self._batch_bindings
        direction, distance, delta = self._direction, self._distance, self._delta
        solid, faces = self._solid, self._faces
        initial_serialization = self._initial_serialization
        derivations, zero_faces = self._derivations, self._zero_faces
        return dict(
            method='unassembled_all_boundary_triangle_prism_union_with_summed_scene_overlap_affine_v3',
            representation='Mathematical union certificate; components are not Boolean-unioned. Expanded prisms are conservative bounds.',
            source=bindings[0], base_source=bindings[1], required_source_bindings=bindings,
            units='mm', direction=direction.tolist(), distance_mm=distance, translation_mm=delta.tolist(),
            starting_oriented_triangles_sha256=B._oriented_triangles_sha(solid),
            actual_starting_stock_preserved=True, starting_serialization=initial_serialization,
            boundary_triangle_count=len(faces), convex_prism_count=len(derivations),
            verification_component_count=self._yielded_components, exactly_zero_volume_face_count=len(zero_faces),
            exactly_zero_volume_face_indices=zero_faces,
            zero_volume_proof='Exact rational determinant of binary-float source points and binary-float delta equals zero.',
            near_parallel_nonzero_faces_discarded=0,
            padded_prism_count=sum(row['padding_mm'] > 0. for row in derivations),
            affine_triangular_prism_count=sum(row['construction'] == 'expanded_affine_triangular_prism' for row in derivations),
            conservative_box_face_indices=[row['face_index'] for row in derivations if row['construction'] == 'thin_oriented_box'],
            padded_face_indices=[row['face_index'] for row in derivations if row['padding_mm'] > 0.],
            maximum_nominal_padding_parameter_mm=max((row['padding_mm'] for row in derivations), default=0.),
            maximum_vertex_enclosure_distance_mm=max((row['maximum_vertex_to_ideal_vertex_distance_mm'] for row in derivations), default=0.),
            enclosure_distance_basis='Maximum exact-rational, outward-rounded output-vertex distance to a nearest ideal vertex; diagnostic upper bound, never a collision exemption.',
            padded_volume_included_in_collision_sum=True,
            analytical_coverage='The starting signed solid plus every boundary triangle swept by delta covers its full continuous translation, including cavity boundaries.',
            actual_component_containment='All six ideal points satisfy every raw and reloaded oriented boundary halfspace exactly. Query components are original convex constructor objects.',
            ideal_endpoint_coordinates='Exact Fraction(source binary-float coordinate) plus Fraction(binary-float delta), without rounding the sum.',
            serialization_uncovered_bound_mm3=0.,
            overlap_rule='SUM every complete-scene component overlap, counting duplicates, plus serialization_uncovered_bound_mm3; accept only total<=1e-8 mm3.',
            prism_derivation_records_sha256=B._sha(json.dumps(derivations, sort_keys=True).encode()),
            derivation_records_scope='Regenerate from the bound actual mesh, translation and both source files; no prior collision proof is reused.',
            raw_and_serialized_exact_halfspace_tests=sum(row['raw_target_containment']['exact_rational_tests']
                + row['serialized_target_containment']['exact_rational_tests'] for row in derivations),
            target_containment_violations=0,
            maximum_conservative_extra_volume_mm3=max((row['conservative_extra_volume_mm3'] for row in derivations), default=0.),
            total_conservative_extra_volume_mm3=sum(row['conservative_extra_volume_mm3'] for row in derivations),
            expected_prism_volume_error_sum_mm3=sum(row['expected_volume_error_mm3'] for row in derivations),
            expected_prism_volume_error_scope='Diagnostic only; literal target containment supplies coverage and all added volume is charged to collisions.',
            literal_roundtrip_triangle_identity_count=sum(row['identical_oriented_triangles'] for row in derivations),
            maximum_roundtrip_uncovered_volume_mm3=max((row['serialization_uncovered_mm3'] for row in derivations), default=0.),
            static_geometry_checks_passed=True, current_complete_scene_check_required=True)


def iter_translation_sweep(solid, direction, distance):
    return TranslationSweepIterator(solid, direction, distance)
