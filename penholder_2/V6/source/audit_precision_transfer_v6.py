#!/usr/bin/env python3
"""Transfer the frozen source motion proof across a bounded export preparation.

This is a scalar proof, not a replacement for strict canonical mesh validation.
The caller must supply a complete source-cache-to-export surface error bound.
No frozen source mesh, motion auditor, or existing envelope is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def derive_motion_transfer(surface_error_mm, pair_allowance_mm=None, *,
                           root=ROOT, added_gap_checks=None, verify_inputs=True):
    """Return a conditional proof for sigma-bounded surface perturbations.

    surface_error_mm is a conservative Euclidean surface displacement from
    the ORIGINAL CERTIFIED LAYER MESHES, including every prior/current
    quantization, simplification and numerical preparation stage. It is not
    merely the last simplify() argument. Added source support material must
    have the same or a smaller bound.

    added_gap_checks may be assembly['all_angle_moving_envelope_checks']; its
    mesh/hash binding remains the caller's responsibility. Returned values
    are conservative lower bounds after the complete perturbation.
    """
    root = Path(root)
    sigma = float(surface_error_mm)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Expected a finite nonnegative complete surface error bound')
    certificate_path = root/'reports/rotation_envelope_certificate.json'
    motion_path = root/'reports/motion_validation.json'
    certificate = json.loads(certificate_path.read_text())
    motion = json.loads(motion_path.read_text())
    if not certificate.get('passed') or not motion.get('passed'):
        raise ValueError('The original source motion proof has not passed')
    manifest = dict(certificate['input_sha256'])
    manifest[str(certificate_path.relative_to(root))] = sha(certificate_path)
    manifest[str(motion_path.relative_to(root))] = sha(motion_path)
    if verify_inputs:
        for name, digest in manifest.items():
            if sha(root/name) != digest:
                raise ValueError('Frozen motion input changed: '+name)
    old_eps_values = {float(m['export_coordinate_allowance_mm']) for m in certificate['modules']}
    if len(old_eps_values) != 1:
        raise ValueError('Expected one declared source export allowance')
    old_eps = old_eps_values.pop()
    if pair_allowance_mm is None:
        # Round upward in 1e-5 mm increments; retain the old allowance when
        # it already covers the complete two-surface perturbation.
        allowance = max(old_eps, math.ceil((2*sigma)/1e-5)*1e-5)
    else:
        allowance = float(pair_allowance_mm)
    if not math.isfinite(allowance) or allowance < old_eps or allowance < 2*sigma:
        raise ValueError('Pair allowance does not cover both perturbed surfaces')
    modules = []
    for basis in certificate['modules']:
        n = basis['layer_count']
        original = next(m for m in motion['modules'] if m['layer_count'] == n)
        joints = []
        layers = []
        bearing_rows = []
        cumulative_theta = 0.
        cumulative_play = 0.
        for old_joint in basis['interfaces']:
            child = old_joint['moving_index']
            pair = [b for b in basis['measured_bearings'] if b['moving_index'] == child]
            if len(pair) != 2 or {b['side'] for b in pair} != {-1,1}:
                raise ValueError('Missing opposed bearing pair')
            radial = max(max(0., b['actual_maximum_bore_mouth_radius_mm']
                             - b['actual_minimum_pin_radius_mm'] + allowance) for b in pair)
            mouths = [b['actual_mouth_axial_mm']-allowance for b in pair]
            if min(mouths) <= 0:
                raise ValueError('Precision budget consumes a bearing lever arm')
            theta = max(math.atan2(max(0., b['actual_maximum_bore_mouth_radius_mm']
                                      - b['actual_minimum_pin_radius_mm'] + allowance), mouth)
                        for b,mouth in zip(pair,mouths))
            axial = max(max(0., b['axial_endplay_mm']) + allowance
                        + (max(0., b['actual_pin_tip_axial_mm'])+allowance)
                        * (1-math.cos(theta)) for b in pair)
            displacement = max(old_joint['centre_displacement_bound_mm'], math.hypot(axial,radial))
            if axial >= min(mouths):
                raise ValueError('Centre may leave the opposed-mouth interval')
            cumulative_theta += theta
            cumulative_play += displacement
            joints.append(dict(host_index=child-1, moving_index=child,
                               radial_center_bound_mm=radial, axial_center_bound_mm=axial,
                               joint_tilt_radians=theta, center_displacement_bound_mm=displacement))
            old_layer = next(l for l in basis['moving_layers'] if l['layer_index'] == child)
            delta_theta = max(0., cumulative_theta-old_layer['cumulative_axis_tilt_radians'])
            delta_play = max(0., cumulative_play-old_layer['cumulative_centre_displacement_mm'])
            padding = (old_layer['maximum_vertex_norm_mm']*delta_theta
                       + delta_play + allowance-old_eps)
            layers.append(dict(layer_index=child, cumulative_tilt_radians=cumulative_theta,
                               cumulative_center_play_mm=cumulative_play,
                               added_axis_cone_radians=delta_theta, added_center_play_mm=delta_play,
                               required_old_envelope_minkowski_radius_mm=padding))
            for b in pair:
                opposite = next(v for v in pair if v['side'] == -b['side'])
                tilt_loss = (2*(b['actual_pin_tip_axial_mm']+sigma)*(1-math.cos(theta))
                             +(b['actual_maximum_pin_radius_mm']+sigma)*math.sin(theta))
                retained = (b['engagement_mm']-opposite['axial_endplay_mm']-4*sigma-tilt_loss)
                radial_gap = b['all_angle_radial_clearance_bound_mm']-2*sigma
                end_gap = b['axial_endplay_mm']-2*sigma
                bearing_rows.append(dict(host_index=b['host_index'], moving_index=child, side=b['side'],
                    running_radial_clearance_lower_bound_mm=radial_gap,
                    axial_endplay_lower_bound_mm=end_gap,
                    retained_engagement_lower_bound_mm=retained,
                    passed=radial_gap>.05 and end_gap>.03 and retained>.3))
        rotation_rows = []
        for prior in original['continuous_rotation']:
            lower = prior['certified_continuous_rotation_gap_mm']-(allowance-old_eps)
            rotation_rows.append(dict(pair=prior['pair'], source_gap_mm=prior['certified_continuous_rotation_gap_mm'],
                transferred_continuous_rotation_gap_mm=lower, passed=lower>0))
        nonadjacent_rows = []
        for prior in original['nonadjacent_shell_clearance']:
            i,j = prior['outer_layer'], prior['inner_layer']
            relative = sum(q['center_displacement_bound_mm'] for q in joints if i<=q['host_index']<j)
            lower = prior['centered_all_angle_spherical_separation_bound_mm']-relative-allowance
            nonadjacent_rows.append(dict(outer_layer=i, inner_layer=j,
                relative_center_displacement_bound_mm=relative,
                transferred_all_angle_separation_lower_bound_mm=lower, passed=lower>0))
        modules.append(dict(layer_count=n, interfaces=joints, moving_layers=layers,
            old_envelope_required_uniform_outer_padding_mm=max(l['required_old_envelope_minkowski_radius_mm'] for l in layers),
            bearings=bearing_rows, adjacent_rotation=rotation_rows, nonadjacent_shells=nonadjacent_rows,
            passed=all(x['passed'] for x in bearing_rows+rotation_rows+nonadjacent_rows)))
    additions = []
    if added_gap_checks is not None:
        for row in added_gap_checks:
            if not row.get('passed') or not row.get('intersection_empty'):
                raise ValueError('A nominal added-material check has not passed')
            n = row['layers']
            padding = next(m['old_envelope_required_uniform_outer_padding_mm'] for m in modules if m['layer_count']==n)
            # The old exact-surface distance can be reduced by at most the
            # envelope expansion and the allowance for added surface/error.
            lower = float(row['computed_envelope_minimum_gap_mm'])-padding-allowance
            additions.append(dict(layers=n, azimuth_deg=row.get('azimuth_deg'),
                old_envelope_outer_padding_mm=padding,
                transferred_continuous_motion_gap_lower_bound_mm=lower,
                source_distance_was_capped=bool(row.get('envelope_distance_capped',False)),
                required_gap_mm=.025, passed=lower>=.025))
    return dict(
        mathematical_transfer_passed=all(m['passed'] for m in modules) and all(a['passed'] for a in additions),
        release_certificate=False,
        condition='Requires a separately validated complete source-to-final surface error bound, strict canonical topology, and final/source mesh hash bindings.',
        per_surface_error_upper_bound_mm=sigma, source_pair_allowance_mm=old_eps,
        effective_pair_allowance_mm=allowance, pair_budget_headroom_mm=allowance-2*sigma,
        source_input_hashes_verified=verify_inputs, source_input_sha256=manifest,
        modules=modules, added_material_gap_transfers=additions,
        proof='For each layer, |cos(u)-cos(v)|<=|u-v| gives h_new(phi)-h_old(phi)<=R*delta_tilt+delta_play+delta_allowance for every support direction. Thus the old certified envelope plus a ball of the reported radius contains the complete perturbed motion. A fixed-pose rotation is an isometry, so each old all-angle surface gap loses at most the complete two-surface error; its already subtracted source allowance is replaced, not counted twice.',
        limitations=[
            'This helper does not establish the caller-supplied error bound or validate the final mesh.',
            'No claim of exactly unchanged exported mechanism geometry is made; immutable source meshes and a bounded representation change are distinct.',
            'Nominal external removal sweeps with zero geometric clearance do not acquire a perturbation-clearance certificate from this motion calculation.',
            'No resin, cure, load or physical removal-force simulation is included.'],
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface-error',type=float,required=True)
    p.add_argument('--pair-allowance',type=float)
    p.add_argument('--assembly-report',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'reports/precision_motion_transfer.json')
    a=p.parse_args()
    assembly=json.loads(a.assembly_report.read_text()) if a.assembly_report else None
    rows=assembly['all_angle_moving_envelope_checks'] if assembly else None
    result=derive_motion_transfer(a.surface_error,a.pair_allowance,added_gap_checks=rows)
    result['auditor_sha256']=sha(__file__)
    if a.assembly_report:result['assembly_report_sha256']=sha(a.assembly_report)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['mathematical_transfer_passed','release_certificate',
        'per_surface_error_upper_bound_mm','effective_pair_allowance_mm','pair_budget_headroom_mm']},indent=2))
    if not result['mathematical_transfer_passed']:raise SystemExit(2)


if __name__=='__main__':main()
