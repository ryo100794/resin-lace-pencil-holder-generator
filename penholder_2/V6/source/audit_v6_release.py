#!/usr/bin/env python3
"""Bind V6 validation results to one immutable set of deliverable inputs.

--geometry-only writes the final-mesh precision/motion transfer before the
slower external-removal and internal-service checks. The full command also
requires both checks and the actual 3MF/STL readback to have passed.
"""
from pathlib import Path
import argparse,hashlib,json,math,zipfile
from audit_precision_transfer_v6 import derive_motion_transfer

ROOT=Path(__file__).resolve().parent.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def gate(name):
    d=json.loads((ROOT/'reports'/name).read_text())
    if d.get('passed') is not True:raise ValueError('Required gate failed: '+name)
    return d

def equal(a,b,label):
    if a!=b:raise ValueError('Mismatched validation binding: '+label)

def verify_manifest(rows):
    for row in rows:
        equal(sha(ROOT/row['path']),row['sha256'],row['path'])

def verify_saved_mesh_file(relative,digest,label):
    path=ROOT/relative
    equal(sha(path),digest,label)
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!={'vertices.npy','faces.npy'} or archive.testzip() is not None:
            raise ValueError('Incomplete or corrupt saved mesh: '+relative)

def check_volumes(values,tolerance,label):
    if any(not isinstance(value,(int,float)) or not math.isfinite(value)
           or value<0 or value>tolerance for value in values):
        raise ValueError('Invalid numerical collision/partition bound: '+label)

def close_number(actual,expected,label):
    if not math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-24):
        raise ValueError('Inconsistent reported bound: '+label)

def path_entries(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():yield from path_entries(child)
    elif isinstance(value,list):
        for child in value:yield from path_entries(child)

def collision_bounds(plans):
    values=[]
    for row in plans:
        values.extend(leg['maximum_swept_intersection_mm3']
            for fragment in row['fragments'] for leg in fragment['translation_legs'])
        endpoint=row.get('endpoint_cut_validation')
        if endpoint is not None:
            values.extend(endpoint['cutter_complete_obstacle_state_overlap_mm3'])
            values.append(endpoint['actual_kerf_complete_obstacle_state_overlap_mm3'])
        if row['mode']=='explicit_body_cuts':values.append(row['kerf_complete_obstacle_state_overlap_mm3'])
    return values

def verify_holder_endpoint_binding(removal,input_hashes,assembly):
    state=removal.get('holder_finite_endpoint_state_binding')
    if not state or state.get('passed') is not True:
        raise ValueError('Missing finite holder endpoint state binding')
    tolerance=removal['numerical_positive_intersection_tolerance_mm3']
    equal(state['unchanged_volume_tolerance_mm3'],tolerance,'Finite holder tolerance')
    if state.get('printed_full_source_geometry_unchanged') is not True:
        raise ValueError('Holder endpoint partition changed printed geometry')
    manifest_path=state['source_manifest'];digest=state['source_manifest_sha256']
    equal(input_hashes.get(manifest_path),digest,'Frozen holder endpoint manifest')
    equal(sha(ROOT/manifest_path),digest,'Actual holder endpoint manifest')
    manifest=json.loads((ROOT/manifest_path).read_text())
    if manifest.get('passed') is not True or manifest.get('printed_geometry_modified') is not False or manifest.get('failures'):
        raise ValueError('Holder finite endpoint partition did not pass')
    equal(manifest['volume_tolerance_mm3'],tolerance,'Holder local partition tolerance')
    for path,source_hash in manifest['inputs'].items():
        equal(input_hashes.get(path),source_hash,'Frozen holder partition input '+path)
    records={row['id']:row for row in manifest['records']}
    runtime={row['id']:row for row in state['records']}
    equal(len(records),len(manifest['records']),'Unique holder partition IDs')
    equal(len(runtime),len(state['records']),'Unique holder runtime IDs')
    equal(set(records),set(runtime),'Holder local/runtime operation set')
    equal(len(records),removal['holder_finite_endpoint_definition_count'],'Holder finite-cut count')
    for identifier,record in records.items():
        if (record.get('passed') is not True or record.get('all_oriented_components_preserved') is not True
            or record.get('serialized_oriented_triangles_identical') is not True):
            raise ValueError('Unverified finite holder partition '+identifier)
        check_volumes(record['checks'].values(),tolerance,identifier+' local partition')
        check_volumes(runtime[identifier]['checks'].values(),tolerance,identifier+' actual state')
        equal(runtime[identifier]['source_full_sha256'],record['source_full_sha256'],'Unchanged full stock '+identifier)
        for entry in path_entries(record['paths']):
            equal(input_hashes.get(entry['path']),entry['sha256'],'Frozen holder cut asset '+entry['path'])
            verify_saved_mesh_file(entry['path'],entry['sha256'],'Holder cut asset '+entry['path'])
    pre_cut_excess=sum(row['checks']['old_cap_outside_unchanged_full_mm3']+
        row['checks']['new_cap_outside_unchanged_full_mm3'] for row in state['records'])
    maximum_excess=sum(row['checks']['new_cap_outside_final_maximum_retained_mm3'] for row in state['records'])
    new_cap_excess=sum(row['checks']['new_cap_outside_unchanged_full_mm3'] for row in state['records'])
    close_number(state['total_pre_cut_occupied_state_excess_bound_mm3'],pre_cut_excess,'Pre-cut occupied-state excess')
    close_number(state['total_new_caps_outside_final_maximum_retained_mm3'],maximum_excess,'Maximum-retained excess')
    check_volumes([pre_cut_excess,maximum_excess],tolerance,'Combined holder state bounds')
    equal(state['final_maximum_retained_path'],'cache/added_retained_v6.npz','Holder maximum retained path')
    equal(state['final_maximum_retained_sha256'],assembly['retained_geometry_sha256'],'Holder/service maximum retained binding')
    prior=None
    continuation=removal.get('continuation_proof')
    if continuation:
        prior_path=continuation['immutable_prior_report']
        equal(input_hashes.get(prior_path),continuation['prior_sha256'],'Frozen prior state proof')
        prior=json.loads((ROOT/prior_path).read_text())
    prior_values=collision_bounds(prior['plans']) if prior else []
    check_volumes(prior_values,tolerance,'Reused prior collision bounds')
    same=bool(prior and prior.get('holder_finite_endpoint_state_binding',{}).get('source_manifest_sha256')==digest)
    applied=0. if same or prior is None else new_cap_excess
    maximum_prior=max(prior_values,default=0.)
    transfer=state['prior_collision_margin_transfer']
    if transfer.get('passed') is not True:raise ValueError('Holder prior-state transfer did not pass')
    equal(transfer['reused_prior_collision_bound_count'],len(prior_values),'Prior collision bound count')
    equal(transfer['same_frozen_definition_already_present_in_prior'],same,'Prior finite-cut definition identity')
    for key,expected in [('maximum_prior_collision_bound_mm3',maximum_prior),
        ('total_new_cap_outside_active_full_volume_mm3',new_cap_excess),
        ('added_obstacle_volume_bound_applied_mm3',applied),
        ('resulting_maximum_prior_bound_mm3',maximum_prior+applied),
        ('remaining_collision_margin_mm3',tolerance-maximum_prior-applied)]:
        close_number(transfer[key],expected,key)
    check_volumes([maximum_prior+applied],tolerance,'Transferred prior collision bound')
    expected_active=set(records) if prior is None else set(records)&{row['id'] for row in prior['remaining_failures']}
    expected_accepted=set() if prior is None else set(records)&{row['id'] for row in prior['plans']}
    equal(set(state['prior_reuse_scope']['still_active_ids']),expected_active,'Still-active holder definitions')
    equal(set(state['prior_reuse_scope']['already_accepted_with_same_definition']),expected_accepted,'Previously accepted holder definitions')
    for row in removal['plans']:
        if row['id'] not in records:continue
        record=records[row['id']];partition=row.get('endpoint_cut_partition') or {}
        equal(partition.get('source_report'),manifest_path,'Holder cut source report')
        equal(partition.get('source_manifest_sha256'),digest,'Holder cut source manifest hash')
        equal(partition.get('paths'),record['paths'],'Holder cut saved geometry')
        equal(partition.get('checks'),record['checks'],'Holder cut local proof')
        equal(row.get('endpoint_cut_bands'),record['cuts'],'Holder cut bands')
        equal(partition.get('source_constructor_sha256'),manifest['source_sha256'],'Holder cut constructor')
        equal(input_hashes.get(partition.get('source_constructor')),manifest['source_sha256'],'Frozen holder cut constructor')
        equal(row.get('holder_endpoint_printed_full_sha256'),record['source_full_sha256'],'Unchanged printed holder stock')
        check_volumes(row['holder_endpoint_runtime_partition_checks'].values(),tolerance,'Accepted holder runtime partition')
    return manifest_path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry-only',action='store_true')
    args=p.parse_args()
    assembly=gate('custom_support_assembly.json')
    build=gate('custom_supports_validation.json')
    prep=gate('v6_precision_preparation.json')
    mechanism=gate('v5_mechanism_unchanged.json')
    verify_manifest(assembly['inputs'])
    for row in mechanism['checks']:
        equal(sha(ROOT/'cache/spheres'/row['file']),row['sha256'],'Original sphere '+row['file'])
    final_path=ROOT/'cache/print_ready_v6.npz';final_hash=sha(final_path)
    equal(assembly['exported_mesh_sha256'],final_hash,'assembly -> final mesh')
    equal(build['input_sha256'],final_hash,'whole-height -> final mesh')
    equal(prep['exported_mesh_sha256'],final_hash,'precision -> final mesh')
    equal(build['auditor_sha256'],sha(ROOT/'source/audit_custom_supports.py'),'whole-height auditor')
    equal(assembly['precision_preparation_report_sha256'],sha(ROOT/'reports/v6_precision_preparation.json'),'precision report')
    equal(assembly['raw_component_diagnostic_sha256'],sha(ROOT/'reports/v6_raw_component_diagnostic.json'),'raw boundary diagnostic')
    equal(prep['source_union_sha256'],sha(ROOT/'cache/assembled_v6_float64.npz'),'raw float64 union')
    equal(prep['simplified_float64_sha256'],sha(ROOT/'cache/assembled_v6_prepared_float64.npz'),'bounded preparation')
    equal(prep['preparation_source_sha256'],sha(ROOT/'source/prepare_v6_export.py'),'preparation implementation')
    equal(prep['canonical_validator_sha256'],sha(ROOT/'source/build_test_coupon.py'),'strict topology implementation')
    equal(assembly['added_support_mesh_sha256'],sha(ROOT/'cache/added_supports_v6.npz'),'actual additions')
    equal(assembly['retained_geometry_sha256'],sha(ROOT/'cache/added_retained_v6.npz'),'maximum retained state')
    topology=assembly['export_mesh_validation']
    if not topology.get('passed') or topology['components']!=1 or topology['bad_edge_incidence_count'] or topology['nonmanifold_vertex_count']:
        raise ValueError('Strict final surface validation failed')
    if build['positive_non_plate_material_birth_count']!=0:
        raise ValueError('Actual final mesh still contains unsupported material births')
    chain=prep['source_chain']
    for name,digest in chain['source_lineage_sha256'].items():equal(sha(ROOT/name),digest,name)
    sigma=(chain['prior_per_surface_rounding_upper_bound_mm']
        +prep['effective_simplification_surface_displacement_bound_mm']
        +prep['final_single_float32_rounding_error_mm'])
    if not math.isclose(sigma,prep['complete_source_to_export_surface_error_upper_bound_mm'],rel_tol=1e-14,abs_tol=1e-18):
        raise ValueError('Incomplete or inconsistent source-to-final precision chain')
    transfer=derive_motion_transfer(sigma,root=ROOT,
        added_gap_checks=assembly['all_angle_moving_envelope_checks'])
    if not transfer['mathematical_transfer_passed']:raise ValueError('Final precision motion transfer failed')
    transfer.update(passed=True,release_certificate=True,
        condition='Complete documented source-to-final error chain and strict final topology/build-order gates verified and hash-bound.',
        exported_mesh_sha256=final_hash,
        assembly_report_sha256=sha(ROOT/'reports/custom_support_assembly.json'),
        precision_preparation_report_sha256=sha(ROOT/'reports/v6_precision_preparation.json'),
        whole_height_report_sha256=sha(ROOT/'reports/custom_supports_validation.json'),
        transfer_source_sha256=sha(ROOT/'source/audit_precision_transfer_v6.py'),
        release_binding_source_sha256=sha(__file__))
    target=ROOT/'reports/v6_precision_motion_validation.json'
    target.write_text(json.dumps(transfer,indent=2,allow_nan=False)+'\n')
    if args.geometry_only:
        print(json.dumps(dict(passed=True,scope='Final-mesh precision and motion binding',
            source_to_final_surface_error_upper_bound_mm=sigma)),flush=True)
        return
    removal=gate('v6_external_removal_validation.json')
    service=gate('v6_internal_service_validation.json')
    exported=gate('v6_export_validation.json')
    verify_manifest(removal['input_files'])
    removal_input_hashes={entry['path']:entry['sha256'] for entry in removal['input_files']}
    if removal['planned_support_count']!=removal['initial_support_count'] or removal['remaining_failures']:
        raise ValueError('External removal is incomplete')
    if len(removal['plans'])!=removal['planned_support_count']:
        raise ValueError('External removal row count differs from its declared count')
    if len({row['id'] for row in removal['plans']})!=removal['initial_support_count']:
        raise ValueError('External removal IDs are duplicated or incomplete')
    tolerance=removal['numerical_positive_intersection_tolerance_mm3']
    check_volumes(collision_bounds(removal['plans']),tolerance,'Complete accepted cut and sweep sequence')
    holder_partition_path=verify_holder_endpoint_binding(removal,removal_input_hashes,assembly)
    for index,row in enumerate(removal['plans']):
        if row.get('passed') is not True:raise ValueError('Unverified external removal row: '+row['id'])
        equal(row['sequence_index'],index,'External removal sequence')
        if row.get('endpoint_cut_bands'):
            check=row.get('endpoint_cut_validation') or {}
            if check.get('passed') is not True or check.get('finite_cutters_checked') is not True:
                raise ValueError('Finite endpoint cuts did not pass: '+row['id'])
            equal(len(check['cutter_complete_obstacle_state_overlap_mm3']),len(row['endpoint_cut_bands']),
                'Endpoint band/cutter count')
        for fragment in row['fragments']:
            verify_saved_mesh_file(fragment['mesh_file'],fragment['sha256'],
                'Actual saved removal fragment '+fragment['mesh_file'])
        kerf=row.get('actual_body_kerf')
        if kerf:
            verify_saved_mesh_file(kerf['path'],kerf['sha256'],
                'Actual saved body kerf '+kerf['path'])
        partition=row.get('finishing_stock_partition')
        if partition:
            path=partition['manifest_path']
            equal(removal_input_hashes.get(path),partition['sha256'],
                'Finishing stock proof is a frozen routing input')
            equal(sha(ROOT/path),partition['sha256'],'Actual finishing stock proof')
            proof=json.loads((ROOT/path).read_text())
            if proof.get('passed') is not True or proof.get('schema')!='plate_stock_partition_v1':
                raise ValueError('Invalid actual-added-stock finishing proof: '+row['id'])
            equal(proof['policy']['id'],row['id'],'Finishing proof support identity')
            equal(proof['policy']['tolerance_mm3'],removal['numerical_positive_intersection_tolerance_mm3'],
                'Finishing stock uses the unchanged numerical gate')
            for group in ['caller_binding_checks','checks']:
                if any(not isinstance(value,(int,float)) or not math.isfinite(value)
                       or value<0 or value>proof['policy']['tolerance_mm3']
                       for value in proof[group].values()):
                    raise ValueError('Finishing stock/state identity gate failed: '+row['id'])
            if proof['policy'].get('method')=='B2_literal_serialized_R_after_exact_empty_whole_original_intersection':
                predicates=proof['exact_measured_predicates']
                for key in ['R_intersection_whole_original_is_empty','R_outside_F_is_empty']:
                    if predicates.get(key) is not True:
                        raise ValueError('Literal finishing stock does not meet the exact empty-set condition: '+row['id'])
                for key in ['R_intersection_whole_original_triangle_count','R_outside_F_triangle_count']:
                    equal(predicates[key],0,'Literal finishing stock has no intersection triangles')
                for key in ['measured_R_intersection_whole_original_mm3','measured_R_outside_F_mm3']:
                    equal(proof['checks'][key],0.,'Literal finishing stock has measured empty-set volume')
                identity=proof['literal_stock_identity'];caller=proof['caller_binding_identity']
                for key in ['cached_C_bytes_equal_original_R','returned_stock_is_same_caller_R_object']:
                    if identity.get(key) is not True:raise ValueError('Literal finishing stock identity failed: '+row['id'])
                for key in ['full_triangle_identity','removable_triangle_identity']:
                    if caller.get(key) is not True:raise ValueError('Caller/source triangle identity failed: '+row['id'])
                equal(caller['full_source_oriented_triangles_sha256'],caller['caller_full_oriented_triangles_sha256'],
                    'Literal finishing full-part triangle identity')
                equal(caller['original_R_oriented_triangles_sha256'],caller['caller_R_oriented_triangles_sha256'],
                    'Literal finishing removable triangle identity')
                original_removable_hash=proof['bindings']['original_removable']['sha256']
                equal(identity['original_R_sha256'],original_removable_hash,'Literal R source hash')
                equal(identity['cached_C_sha256'],original_removable_hash,'Literal C source-byte hash')
                cleaned=next(entry for entry in proof['files'] if entry['role']=='cleaned_removable')
                equal(cleaned['sha256'],original_removable_hash,'Literal C cache-byte hash')
                for key in ['numerical_CSG_identity_tests_performed','unexecuted_CSG_tests_reported_as_numeric_zero']:
                    if proof['algebraic_state_proof'].get(key) is not False:
                        raise ValueError('Literal-stock algebraic/numerical evidence is not distinguished: '+row['id'])
            elif proof['policy'].get('method')=='B2_wraps_unchanged_frozen_B_clipped_stock':
                delegation=proof['delegation_proof'];delegated=proof['bindings']['delegated_partition']
                equal(delegation['frozen_B_manifest'],delegated,'Delegated finishing proof identity')
                if delegation.get('stock_and_original_B_checks_unchanged') is not True:
                    raise ValueError('Delegated finishing stock was changed: '+row['id'])
                original_proof=json.loads((ROOT/delegated['path']).read_text())
                if original_proof.get('passed') is not True:
                    raise ValueError('Delegated finishing proof did not pass: '+row['id'])
                equal(original_proof['policy']['id'],row['id'],'Delegated finishing support identity')
                for key in ['checks','caller_binding_checks','serialization_checks']:
                    equal(proof[key],original_proof[key],'Unchanged delegated finishing evidence: '+key)
                for entry in original_proof['files']:
                    if entry not in proof['files']:
                        raise ValueError('Delegated finishing stock file omitted: '+entry['path'])
            for entry in list(proof['bindings'].values())+proof['files']:
                equal(removal_input_hashes.get(entry['path']),entry['sha256'],
                    'Finishing stock input/derived file is frozen: '+entry['path'])
            equal(proof['bindings']['original']['path'],'cache/original_v5.npz',
                'Finishing stock preserves the assembled original')
            if not partition.get('original_material_preserved') or not partition.get('physical_retained_state_unchanged'):
                raise ValueError('Finishing stock would change the retained original: '+row['id'])
    equal(service['retained_state_sha256'],assembly['retained_geometry_sha256'],'internal service retained state')
    if 'input_files' in service:verify_manifest(service['input_files'])
    equal(exported['input_sha256'],final_hash,'file export source')
    for data in exported['files']:
        name=data['name']
        equal(sha(ROOT/'deliverables'/name),data['sha256'],'Actual delivered '+name)
    report_names=['custom_support_assembly.json','custom_supports_validation.json',
        'v6_precision_preparation.json','v6_precision_motion_validation.json',
        'rotation_envelope_certificate.json','v6_external_removal_validation.json',
        'v6_internal_service_validation.json','v6_export_validation.json','v5_mechanism_unchanged.json']
    report_names.append(Path(holder_partition_path).name)
    result=dict(passed=True,version=6,final_mesh_sha256=final_hash,
        reports=[dict(path='reports/'+name,sha256=sha(ROOT/'reports'/name)) for name in report_names],
        release_auditor_sha256=sha(__file__),
        positive_non_plate_material_birth_count=0,
        strict_closed_connected_surface=True,
        external_removal_geometry_scope='Nominal cut planes, declared kerfs and explicitly partitioned actual added underside finishing stock; continuous swept bounds with complete remaining obstacle state. A finishing-stock partition does not certify a physical cutter along its irregular contact surface.',
        internal_service_geometry_scope='Maximum retained new stumps and all new permanent additions checked against original internal tool/extraction paths.',
        exported_representation_scope='A documented bounded simplification and float32 rounding; motion clearance transferred across the complete error chain.',
        physical_print_tested=False,
        limitations=['No printer, resin, cure, peeling-load or physical break-force test.',
            'External cut/removal paths use nominal source geometry. Zero-clearance cut boundaries are not certified against arbitrary manufacturing or numerical perturbations.',
            'Full cutter bodies, handles and finger access are not simulated.'])
    (ROOT/'reports/v6_release_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,final_mesh_sha256=final_hash,
        external_parts=removal['planned_support_count'],physical_print_tested=False)),flush=True)

if __name__=='__main__':main()
