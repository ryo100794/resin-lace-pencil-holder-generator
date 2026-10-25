#!/usr/bin/env python3
"""Literal-stock no-op proof for the 24 original-disjoint underside pads.

Frozen B remains the owner of the exact/conservative sweep implementation
and of the 16 previously validated clipped-stock partitions.  This separate
helper does not modify B, E, printed geometry or an existing cache.

For the declared 24-pad subset, direct intersection with the COMPLETE
original must be literally empty, and R minus F must also be empty. Then
C=R-original=R is represented by the same caller object and a byte-for-byte
copy of the original serialized R. The old retained K=F-R is unchanged.
The retained-state and partition identities are algebraic consequences of
those literal assignments; they are NOT presented as numerical CSG tests.

The public prepare_plate_stock API and schema remain compatible with B.
Both source files must appear in the caller's frozen input hash ledger.
"""
from pathlib import Path
import hashlib
import json

import manifold3d as mf
import external_removal_plate_stock_v6 as B

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve()
TOL = B.TOL
PLATE_IDS = B.PLATE_IDS
NOOP_PLATE_IDS = frozenset(sid for sid in PLATE_IDS if int(sid[1:]) < 579)
CertificateError = B.CertificateError
translation_sweep_components = B.translation_sweep_components
verify_translation = B.verify_translation
FROZEN_B_SHA256 = 'bdcc40203f31000d8918066de93de0a525e9cc493f7c4ceb29c13f631b11b019'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bindings(item, original_entry, expected):
    source, _ = B._entry(SOURCE, expected)
    frozen_b, _ = B._entry(B.__file__, expected)
    if frozen_b['sha256'] != FROZEN_B_SHA256:
        raise CertificateError('B2 requires the reviewed, immutable B implementation')
    original = dict(path=B._relative(original_entry['path']),
                    sha256=original_entry['sha256'])
    if expected.get(original['path']) != original['sha256']:
        raise CertificateError('Original scene is not bound in the caller ledger')
    folder = ROOT/'cache/custom_holder_supports'
    full_entry, full_bytes = B._entry(folder/item['record']['full_mesh_file'], expected)
    rem_entry, rem_bytes = B._entry(folder/item['record']['removable_mesh_file'], expected)
    bindings = dict(source=source, exact_sweep_source=frozen_b, original=original,
                    full=full_entry, original_removable=rem_entry)
    return bindings, full_bytes, rem_bytes


def _cache_location(item, bindings, policy, cache_root):
    digest = _sha(json.dumps(dict(bindings=bindings, policy=policy), sort_keys=True).encode())
    base = (Path(cache_root) if cache_root is not None
            else ROOT/'cache/external_removal_plate_stock_v6_b2')
    if not base.is_absolute():
        base = ROOT/base
    folder = base/item['id']/digest
    return folder, folder/'partition.json'


def _read_cached(manifest_path, bindings, policy):
    if not manifest_path.exists():
        return None
    record = json.loads(manifest_path.read_bytes())
    if (record.get('passed') is not True or record.get('bindings') != bindings
            or record.get('policy') != policy):
        raise CertificateError('Invalid immutable B2 cache binding')
    for entry in record['files']:
        if _sha((ROOT/entry['path']).read_bytes()) != entry['sha256']:
            raise CertificateError('Derived B2 cache file changed: '+entry['path'])
    return record


def _delegate_clipped(item, original, original_entry, expected, bindings, cache_root):
    # Reuse the existing B cache without changing it. The wrapping record is
    # separately bound to B2 so the new caller need not infer its provenance.
    prepared = B.prepare_plate_stock(
        item, original, original_entry, expected,
        cache_root=ROOT/'cache/external_removal_plate_stock_v6')
    old = prepared['record']
    if old.get('passed') is not True:
        raise CertificateError('Delegated frozen-B partition has not passed')
    old_path = Path(prepared['manifest_path'])
    old_entry = dict(path=B._relative(old_path), sha256=_sha(old_path.read_bytes()))
    bindings = dict(bindings, delegated_partition=old_entry)
    policy = dict(id=item['id'], tolerance_mm3=TOL,
                  method='B2_wraps_unchanged_frozen_B_clipped_stock')
    _, manifest_path = _cache_location(item, bindings, policy, cache_root)
    cached = _read_cached(manifest_path, bindings, policy)
    if cached is not None:
        return dict(stock=prepared['stock'], record=cached, manifest_path=str(manifest_path))
    record = dict(old)
    record.update(bindings=bindings, policy=policy,
                  delegation_proof=dict(frozen_B_manifest=old_entry,
                                        stock_and_original_B_checks_unchanged=True))
    record['files'] = list(old['files']) + [dict(role='frozen_B_partition_manifest', **old_entry)]
    B._immutable_write(manifest_path, (json.dumps(record, indent=2, sort_keys=True)+'\n').encode())
    return dict(stock=prepared['stock'], record=record, manifest_path=str(manifest_path))


def prepare_plate_stock(item, original, original_entry,
                        expected_input_hashes, cache_root=None):
    """Return unchanged literal R only after exact whole-original emptiness.

    Numerical checks contain measured values only. Boolean hypotheses and
    literal/algebraic identities are explicit separate manifest fields.
    The caller keeps its original full-part and old permanent obstacles.
    """
    if item.get('id') not in PLATE_IDS or item.get('kind') != 'plate_perimeter_tack':
        raise CertificateError('Part is outside the declared 40-pad scope')
    B._valid(original, 'Caller-bound whole original')
    bindings, full_bytes, rem_bytes = _bindings(item, original_entry, expected_input_hashes)
    if item['id'] not in NOOP_PLATE_IDS:
        return _delegate_clipped(item, original, original_entry, expected_input_hashes,
                                 bindings, cache_root)

    # This comparison uses literal oriented triangles, not subtraction of
    # two coincident copies. All signed boundaries and cavities are kept.
    source_full = B._load_bytes(full_bytes)
    source_rem = B._load_bytes(rem_bytes)
    full_source_triangles = B._oriented_triangles_sha(source_full)
    rem_source_triangles = B._oriented_triangles_sha(source_rem)
    caller_full_triangles = B._oriented_triangles_sha(item['full'])
    caller_rem_triangles = B._oriented_triangles_sha(item['removable'])
    if full_source_triangles != caller_full_triangles or rem_source_triangles != caller_rem_triangles:
        raise CertificateError('Caller F/R do not literally match their serialized source triangles')

    # These are the only geometric predicates needed for the no-op proof.
    # They are measured against the whole original, not a substitute crop.
    clean = item['removable']
    overlap = clean ^ original
    outside_full = clean - item['full']
    B._valid(overlap, 'Direct R intersect whole original')
    B._valid(outside_full, 'Direct R minus F')
    checks = dict(measured_R_intersection_whole_original_mm3=float(overlap.volume()),
                  measured_R_outside_F_mm3=float(outside_full.volume()))
    predicates = dict(
        R_intersection_whole_original_is_empty=overlap.is_empty(),
        R_intersection_whole_original_triangle_count=overlap.num_tri(),
        R_outside_F_is_empty=outside_full.is_empty(),
        R_outside_F_triangle_count=outside_full.num_tri())
    if not overlap.is_empty() or not outside_full.is_empty():
        raise CertificateError('Literal no-op requires empty sets, not merely small volume',
                               dict(measured_checks=checks, exact_predicates=predicates))
    B._check_gate(checks, 'Measured no-op predicates failed')
    policy = dict(id=item['id'], tolerance_mm3=TOL,
                  method='B2_literal_serialized_R_after_exact_empty_whole_original_intersection')
    folder, manifest_path = _cache_location(item, bindings, policy, cache_root)
    cached = _read_cached(manifest_path, bindings, policy)
    if cached is not None:
        c_entry = next(entry for entry in cached['files'] if entry['role']=='cleaned_removable')
        if c_entry['sha256'] != bindings['original_removable']['sha256']:
            raise CertificateError('Cached no-op C is not byte-identical to original R')
        return dict(stock=clean, record=cached, manifest_path=str(manifest_path))

    # Copy R bytes exactly; no redundant reconstruction/difference changes
    # the adopted stock. The intersection already measured above is empty.
    c_path = folder/'cleaned_removable.npz'
    excluded_path = folder/'excluded_original_stock.npz'
    excluded_bytes = B._bytes(overlap)
    excluded_loaded = B._load_bytes(excluded_bytes)
    if not excluded_loaded.is_empty():
        raise CertificateError('Empty excluded stock changed during serialization')
    B._immutable_write(c_path, rem_bytes)
    B._immutable_write(excluded_path, excluded_bytes)
    if c_path.read_bytes() != rem_bytes:
        raise CertificateError('Literal R cache copy changed bytes')
    files = [dict(role='cleaned_removable', path=B._relative(c_path), sha256=_sha(rem_bytes)),
             dict(role='excluded_original_stock', path=B._relative(excluded_path),
                  sha256=_sha(excluded_bytes))]
    record = dict(
        schema='plate_stock_partition_v1', passed=True, units='mm',
        bindings=bindings, policy=policy, files=files,
        caller_binding_checks={}, checks=checks,
        caller_binding_identity=dict(
            proof_basis='Exact oriented-triangle hashes and source byte hashes; no coincident-source CSG subtraction.',
            full_source_oriented_triangles_sha256=full_source_triangles,
            caller_full_oriented_triangles_sha256=caller_full_triangles,
            original_R_oriented_triangles_sha256=rem_source_triangles,
            caller_R_oriented_triangles_sha256=caller_rem_triangles,
            full_triangle_identity=True, removable_triangle_identity=True),
        exact_measured_predicates=predicates,
        literal_stock_identity=dict(
            returned_stock_is_same_caller_R_object=clean is item['removable'],
            cached_C_bytes_equal_original_R=True,
            original_R_sha256=bindings['original_removable']['sha256'],
            cached_C_sha256=_sha(rem_bytes)),
        algebraic_state_proof=dict(
            basis='Measured R intersect O is EMPTY and R minus F is EMPTY; C is literal R and E is the measured empty intersection.',
            partition='C union E = R because C=R and E=empty; their interiors are disjoint.',
            physical_original_preservation='C intersect O is the measured empty R intersect O.',
            retained_state='F minus C and F minus R are the same expression on the same objects; existing K remains unchanged.',
            complete_scene='(O union F) minus C = O union (F minus R) because C=R and R intersect O=empty.',
            numerical_CSG_identity_tests_performed=False,
            unexecuted_CSG_tests_reported_as_numeric_zero=False),
        volumes_mm3=dict(original_removable=float(clean.volume()),
                         cleaned_removable=float(clean.volume()),
                         preserved_original_overlap=float(overlap.volume())),
        static_obstacle_rule='Keep original O, actual full F and old permanent K=F-R unchanged; only the logical no-op C=R reference is supplied.',
        original_binding='Direct complete-original intersection using caller already hash-bound original object; no duplicate scene load.',
        cut_method='Underside finishing of actual added stock; no original material is in this literal removal stock.',
        cut_boundary_basis='Actual serialized R surface; no historical trim cylinder reconstructed.',
        finite_cutter_or_manual_accuracy_certified=False,
        full_scene_route_validation='Required separately against the current complete obstacle state.')
    B._immutable_write(manifest_path, (json.dumps(record, indent=2, sort_keys=True)+'\n').encode())
    return dict(stock=clean, record=record, manifest_path=str(manifest_path))
