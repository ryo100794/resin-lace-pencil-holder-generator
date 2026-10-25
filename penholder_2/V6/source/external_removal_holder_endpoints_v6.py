#!/usr/bin/env python3
"""Install finite endpoint cuts for eight unchanged, still-active holder supports.

Only removal definitions change. The printed full solids remain byte-identical.
On first adoption, each affected full support must still be present throughout
the reused certified prefix. Both old and new retained caps are contained in
that full support, so its pre-cut occupied state is unchanged. Subsequent reuse
requires the same explicit partition to be bound by the prior accepted row.
"""
from pathlib import Path
import copy
import json
import math

import plan_external_removal_v6 as P

MANIFEST = 'reports/holder_finite_endpoint_partition_v6.json'
CONSTRUCTOR = 'source/holder_finite_endpoint_partition_v6.py'
AFFECTED = {'H0206', 'H0226', 'H0267', 'H0277', 'H0384', 'H0492', 'H0550', 'H0613'}
_INSTALLED = False
_MANIFEST_DATA = None
_MANIFEST_SHA = None
_RECORDS = {}
_PRIOR = None
_BASE_LOAD = None
_BASE_SAVE = None
_STATE_BINDING = None
_SWEEP_BINDINGS = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def small(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= P.TOL


def derived(entry):
    result = P.solid(P.ROOT / entry['path'])
    require(P.INPUT_HASHES[entry['path']] == entry['sha256'],
            'Changed finite holder-cut derived mesh: ' + entry['path'])
    return result


def _load_definition():
    global _MANIFEST_DATA, _MANIFEST_SHA, _RECORDS
    if _MANIFEST_DATA is not None:
        return
    report = P.metadata(P.ROOT / MANIFEST)
    require(report.get('passed') is True and not report.get('failures'),
            'Holder finite endpoint partition has not passed')
    require(report.get('volume_tolerance_mm3') == P.TOL and report.get('printed_geometry_modified') is False,
            'Holder finite-cut scope or tolerance differs')
    P.frozen_bytes(P.ROOT / CONSTRUCTOR)
    require(P.INPUT_HASHES[CONSTRUCTOR] == report['source_sha256'], 'Finite holder-cut constructor changed')
    for path, digest in report['inputs'].items():
        P.frozen_bytes(P.ROOT / path)
        require(P.INPUT_HASHES[path] == digest, 'Finite holder-cut source changed: ' + path)
    records = {row['id']: row for row in report['records']}
    require(len(records) == len(report['records']) and set(records) == AFFECTED,
            'Unexpected finite holder-cut source set')
    for identifier, row in records.items():
        require(row.get('passed') is True and row.get('serialized_oriented_triangles_identical') is True,
                'Unverified serialized holder partition: ' + identifier)
        require(row.get('all_oriented_components_preserved') is True,
                'A holder partition discarded an oriented component')
        require(all(small(x) for x in row['checks'].values()), 'Finite holder partition failed: ' + identifier)
        require(len(row['cuts']) == 2 and len(row['paths']['cutters']) == 2,
                'Holder endpoint cut count differs')
    _MANIFEST_DATA, _MANIFEST_SHA, _RECORDS = report, P.INPUT_HASHES[MANIFEST], records


def verify_prior_definition(prior):
    """Called only after the original continuation has verified every prior byte."""
    _load_definition()
    accepted = {row['id']: row for row in prior['plans']}
    active = {row['id'] for row in prior['remaining_failures']}
    require(AFFECTED <= set(accepted) | active, 'Finite-cut support missing from prior source set')
    for identifier in AFFECTED:
        if identifier in active:
            continue
        row = accepted[identifier]
        partition = row.get('endpoint_cut_partition') or {}
        record = _RECORDS[identifier]
        require(partition.get('source_report') == MANIFEST and
                partition.get('source_manifest_sha256') == _MANIFEST_SHA and
                partition.get('paths') == record['paths'] and
                row.get('endpoint_cut_bands') == record['cuts'],
                'Cannot retroactively change an already removed holder: ' + identifier)
        checks = row.get('endpoint_cut_validation') or {}
        require(checks.get('passed') is True and checks.get('finite_cutters_checked') is True,
                'Prior holder endpoint cuts were not checked')
        require(all(small(x) for x in checks['cutter_complete_obstacle_state_overlap_mm3']) and
                small(checks['actual_kerf_complete_obstacle_state_overlap_mm3']),
                'Prior holder finite cuts intersected the complete state')
    old_binding = prior.get('holder_finite_endpoint_state_binding')
    if old_binding is not None:
        require(old_binding.get('passed') is True and old_binding.get('source_manifest_sha256') == _MANIFEST_SHA,
                'Prior holder endpoint state is bound to a different definition')
    return dict(still_active_ids=sorted(AFFECTED & active),
                already_accepted_with_same_definition=sorted(AFFECTED & set(accepted)))


def load_parts():
    global _STATE_BINDING
    items, permanent, inputs, all_full = _BASE_LOAD()
    by_id = {item['id']: item for item in items}
    require(AFFECTED <= set(by_id), 'Finite-cut holder source operation disappeared')
    holder_path = P.ROOT / 'cache/custom_holder_supports/holder_supports.json'
    if not holder_path.exists():
        holder_path = P.ROOT / 'reports/custom_holder_supports.json'
    holder = P.metadata(holder_path)
    positions = {f'H{row["index"]:04d}': pos for pos, row in enumerate(holder['parts'])}
    maximum = P.solid(P.ROOT / 'cache/added_retained_v6.npz')
    runtime_rows = []
    for identifier in sorted(AFFECTED):
        item = by_id[identifier]
        record = _RECORDS[identifier]
        require(item['group'] == 'holder' and item['source_index'] == record['support_index'],
                'Wrong holder finite-cut identity')
        for path_key, hash_key in [('source_full_mesh_file', 'source_full_sha256'),
                                   ('source_removable_mesh_file', 'source_removable_sha256'),
                                   ('source_maximum_mesh_file', 'source_maximum_sha256')]:
            P.frozen_bytes(P.ROOT / record[path_key])
            require(P.INPUT_HASHES[record[path_key]] == record[hash_key],
                    'Changed finite-cut source stock: ' + record[path_key])
        require(record['source_full_mesh_file'] == 'cache/custom_holder_supports/' + item['record']['full_mesh_file'] and
                record['source_removable_mesh_file'] == 'cache/custom_holder_supports/' + item['record']['removable_mesh_file'],
                'Partition was derived from another printed holder part')
        full = item['full']
        old_cap = permanent[positions[identifier]]
        paths = record['paths']
        cap, body, kerf = [derived(paths[key]) for key in ['retained', 'removable', 'kerf']]
        cutters = [derived(entry) for entry in paths['cutters']]
        fragments = [derived(entry) for entry in paths['fragments']]
        proxies = [derived(entry) for entry in paths['proxies']]
        union = P.H.union([cap, body, kerf])
        checks = dict(
            old_cap_outside_unchanged_full_mm3=P.H.volume(old_cap - full),
            new_cap_outside_unchanged_full_mm3=P.H.volume(cap - full),
            free_body_outside_unchanged_full_mm3=P.H.volume(body - full),
            kerf_outside_unchanged_full_mm3=P.H.volume(kerf - full),
            new_cap_outside_final_maximum_retained_mm3=P.H.volume(cap - maximum),
            partition_uncovered_full_mm3=P.H.volume(full - union),
            partition_outside_full_mm3=P.H.volume(union - full),
            partition_signed_volume_error_mm3=abs(cap.volume() + body.volume() + kerf.volume() - full.volume()),
            retained_body_overlap_mm3=P.H.volume(cap ^ body),
            retained_kerf_overlap_mm3=P.H.volume(cap ^ kerf),
            body_kerf_overlap_mm3=P.H.volume(body ^ kerf),
            fragment_uncovered_body_mm3=P.H.volume(body - P.H.union(fragments)),
            fragment_outside_body_mm3=P.H.volume(P.H.union(fragments) - body),
            proxy_uncovered_body_mm3=P.H.volume(body - P.H.union(proxies)),
            kerf_outside_declared_wafers_mm3=P.H.volume(kerf - P.H.union(cutters)))
        require(all(small(x) for x in checks.values()),
                'Current finite-cut state binding failed for ' + identifier + ': ' + str(checks))
        permanent[positions[identifier]] = cap
        item['_pre_finite_cut_removable'] = item['removable']
        item['removable'], item['pieces'] = body, proxies
        item['endpoint_kerf'], item['endpoint_cutters'] = kerf, cutters
        item['endpoint_partition'], item['precut_fragments'] = record, fragments
        item['_holder_endpoint_runtime_checks'] = checks
        runtime_rows.append(dict(id=identifier, source_full_sha256=record['source_full_sha256'], checks=checks))
    del maximum
    pre_cut_outside_bound = sum(row['checks']['old_cap_outside_unchanged_full_mm3'] +
                                row['checks']['new_cap_outside_unchanged_full_mm3'] for row in runtime_rows)
    new_cap_outside_bound = sum(row['checks']['new_cap_outside_final_maximum_retained_mm3'] for row in runtime_rows)
    require(small(pre_cut_outside_bound) and small(new_cap_outside_bound),
            'Combined finite-cut occupied-state/max-retained bound failed')
    prior_scope = verify_prior_definition(_PRIOR) if _PRIOR is not None else dict(
        still_active_ids=sorted(AFFECTED), already_accepted_with_same_definition=[])
    new_cap_excess = sum(row['checks']['new_cap_outside_unchanged_full_mm3'] for row in runtime_rows)
    prior_bounds = []
    if _PRIOR is not None:
        for row in _PRIOR['plans']:
            for fragment in row['fragments']:
                prior_bounds.extend(leg['maximum_swept_intersection_mm3']
                                    for leg in fragment['translation_legs'])
            endpoint = row.get('endpoint_cut_validation')
            if endpoint is not None:
                prior_bounds.extend(endpoint['cutter_complete_obstacle_state_overlap_mm3'])
                prior_bounds.append(endpoint['actual_kerf_complete_obstacle_state_overlap_mm3'])
            if row.get('mode') == 'explicit_body_cuts':
                prior_bounds.append(row['kerf_complete_obstacle_state_overlap_mm3'])
        require(all(small(value) for value in prior_bounds), 'Invalid prior collision bound')
    maximum_prior_bound = max(prior_bounds, default=0.)
    # A newly introduced cap can add at most its entire outside-F volume
    # to any prior physical sweep or cut. Old caps removed from the scene
    # can only reduce obstruction. When the same caps were already bound,
    # their exact immutable bytes are reused and there is no new material.
    already_same_definition = (_PRIOR is not None and
        _PRIOR.get('holder_finite_endpoint_state_binding', {}).get('source_manifest_sha256') == _MANIFEST_SHA)
    transfer_excess = 0. if already_same_definition or _PRIOR is None else new_cap_excess
    require(maximum_prior_bound + transfer_excess <= P.TOL,
            'New-cap state-transfer excess exhausts a prior collision allowance')
    prior_transfer = dict(
        reused_prior_collision_bound_count=len(prior_bounds),
        maximum_prior_collision_bound_mm3=maximum_prior_bound,
        total_new_cap_outside_active_full_volume_mm3=new_cap_excess,
        added_obstacle_volume_bound_applied_mm3=transfer_excess,
        resulting_maximum_prior_bound_mm3=maximum_prior_bound + transfer_excess,
        remaining_collision_margin_mm3=P.TOL - maximum_prior_bound - transfer_excess,
        same_frozen_definition_already_present_in_prior=already_same_definition,
        passed=True)
    _STATE_BINDING = dict(
        passed=True, source='source/external_removal_holder_endpoints_v6.py',
        source_manifest=MANIFEST, source_manifest_sha256=_MANIFEST_SHA,
        printed_full_source_geometry_unchanged=True,
        prior_reuse_scope=prior_scope, prior_collision_margin_transfer=prior_transfer,
        pre_cut_identity='For every newly adopted definition, its unchanged full support is still active and contains both old and new caps. Thus F union old_cap and F union new_cap have the same occupied state within the stated unchanged bound.',
        total_pre_cut_occupied_state_excess_bound_mm3=pre_cut_outside_bound,
        total_new_caps_outside_final_maximum_retained_mm3=new_cap_outside_bound,
        unchanged_volume_tolerance_mm3=P.TOL,
        final_maximum_retained_path='cache/added_retained_v6.npz',
        final_maximum_retained_sha256=P.INPUT_HASHES['cache/added_retained_v6.npz'],
        records=runtime_rows,
        complete_scene_endpoint_cut_validation_required_for_every_accepted_affected_operation=True)
    print('Finite holder endpoint definitions bound to unchanged full supports and final maximum retained state.', flush=True)
    return items, permanent, inputs, all_full


def save_plan(item, plan, sequence):
    row = _BASE_SAVE(item, plan, sequence)
    if item['id'] in AFFECTED:
        require((row.get('endpoint_cut_validation') or {}).get('passed') is True,
                'Holder finite endpoint cuts were not validated against the complete current scene')
        row['endpoint_cut_partition']['source_report'] = MANIFEST
        row['endpoint_cut_partition']['source_manifest_sha256'] = _MANIFEST_SHA
        row['endpoint_cut_partition']['source_constructor'] = CONSTRUCTOR
        row['endpoint_cut_partition']['source_constructor_sha256'] = P.INPUT_HASHES[CONSTRUCTOR]
        row['holder_endpoint_runtime_partition_checks'] = item['_holder_endpoint_runtime_checks']
        row['holder_endpoint_printed_full_sha256'] = _RECORDS[item['id']]['source_full_sha256']
    return row


def install(prior=None):
    global _INSTALLED, _PRIOR, _BASE_LOAD, _BASE_SAVE, _SWEEP_BINDINGS
    require(not _INSTALLED, 'Holder finite endpoint overlay cannot be installed twice')
    P.frozen_bytes(__file__)
    _load_definition()
    import external_removal_extended_v6 as E
    import external_removal_prisms_v6_b3 as B3
    _SWEEP_BINDINGS = B3.source_bindings()
    for entry in _SWEEP_BINDINGS:
        P.frozen_bytes(P.ROOT / entry['path'])
        require(P.INPUT_HASHES[entry['path']] == entry['sha256'], 'Tight prism helper changed')
    # Replace only E's helper module reference. Frozen B and B2 retain their
    # original functions and source identity; stock preparation is unchanged.
    E.B = B3
    if prior is not None:
        verify_prior_definition(prior)
    _PRIOR = prior
    _BASE_LOAD, _BASE_SAVE = P.load_parts, P.save_plan
    P.load_parts, P.save_plan = load_parts, save_plan
    _INSTALLED = True


def finish_report(path):
    require(_STATE_BINDING is not None and _STATE_BINDING['passed'], 'No holder endpoint state binding')
    report = json.loads(path.read_text())
    report['holder_finite_endpoint_state_binding'] = copy.deepcopy(_STATE_BINDING)
    report['assembly_geometry_binding']['holder_finite_endpoint_state_binding_passed'] = True
    report['holder_finite_endpoint_definition_count'] = len(AFFECTED)
    report['tight_continuous_prism_helper_bindings'] = copy.deepcopy(_SWEEP_BINDINGS)
    report['tight_continuous_prism_scope'] = (
        'Actual signed starting stock and every boundary triangle remain covered; '
        'raw and reloaded exact ideal-vertex halfspace proofs and the unchanged complete-scene summed collision gate remain required. '
        'This changes analysis bounds only, not printed or cut stock.')
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


if __name__ == '__main__':
    # Fresh released rebuild: all supplied instructions are independently
    # replayed; the prefix verifier disables discovery and closes every ID.
    import certify_external_removal_prefix_v6 as R
    base_install, base_finish = R.C.install, R.C.finish_report

    def install_fresh(phase):
        base_install(phase)
        install()

    def finish_fresh(path):
        base_finish(path)
        finish_report(path)

    R.C.install, R.C.finish_report = install_fresh, finish_fresh
    R.main()
