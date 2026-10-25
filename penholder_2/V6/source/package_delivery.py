#!/usr/bin/env python3
"""Create verified, ordinary V5 ZIP archives below the Drive upload limit.

The full-size 3MF is uploaded directly. STL and GLB each get their own
ordinary ZIP; the source project never embeds those large full-holder files.
All ZIPs use the same relative project layout and may be extracted into one
folder. A failed or stale validation stops packaging before promotion.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DELIVERY = ROOT/'deliverables'
CACHE = ROOT/'cache'/'spheres'
STEM = 'ivory_multisphere_pen_holder_v5_printpath_20261025'
COUPON_STEM = 'ivory_sphere_v5_test_coupon'
UPLOAD_LIMIT_BYTES = 104_857_600
REVISION = 'V5 connected print starts'

REQUIRED_REPORTS = (
    'geometry_validation.json', 'motion_validation.json',
    'holder_motion_clearance.json', 'holder_service_access.json',
    'precision_repairs_scope.json', 'early_support_validation.json',
    'strength_validation.json', 'test_coupon_validation.json',
    'full_height_validation.json',
)
REPORT_NAMES = REQUIRED_REPORTS + (
    'design.json', 'print_preparation.json', 'moving_envelopes.json',
    'moving_envelopes_details.json', 'support_projection_clearance.json',
    'sphere_generation_checks.json', 'early_support_v3_baseline.json', 'early_support_v4_baseline.json',
    'full_height_v4_baseline.json',
)
IMAGE_MAP = (
    ('final', 'hero.jpg', 'hero'), ('final', 'top.jpg', 'top'),
    ('final', 'views.jpg', 'views'), ('final', 'detail.jpg', 'detail'),
    ('mechanism', 'mechanism_overview.jpg', 'mechanism'),
    ('mechanism', 'neck_detail.jpg', 'neck_detail'),
    ('mechanism', 'starter_access.jpg', 'starter_access'),
    ('mechanism', 'bearing_section.jpg', 'bearing_section'),
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def check_hash(path, expected, label):
    path = Path(path)
    require(path.is_file(), f'Missing {label}: {path}')
    require(bool(expected) and digest(path) == expected, f'Stale {label}: {path}')


def validate_module_report(report, label):
    modules = report.get('modules', [])
    require(sorted(m['layer_count'] for m in modules) == [3, 4, 5],
            f'{label} does not cover exactly the three requested sphere modules')
    for module in modules:
        name = Path(module['metadata_file']).name
        check_hash(CACHE/name, module['metadata_sha256'], label+' metadata')
        for name, expected in module.get('source_mesh_hashes', {}).items():
            require(Path(name).name == name, 'Unexpected cache reference path')
            check_hash(CACHE/name, expected, label+' source mesh')
        # The motion audit records its actual inputs as topology objects.
        def visit(value):
            if isinstance(value, dict):
                if str(value.get('file', '')).endswith('.npz') and 'sha256' in value:
                    check_hash(CACHE/Path(value['file']).name, value['sha256'], label+' mesh')
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(module)


def validate_exports(report, geometry):
    source = ROOT/'cache'/'print_ready_v5.npz'
    check_hash(source, report.get('source_sha256'), 'export source')
    check_hash(source, geometry.get('sha256'), 'geometry audit source')
    require(report.get('source_triangle_count') == geometry['triangles'],
            'Export and geometry audit triangle counts differ')
    require(report.get('decimation') is False, 'Export reports geometry decimation')
    canonical = report.get('canonical_triangle_sha256')
    require(bool(canonical), 'No canonical export triangle digest')
    files = report.get('files', {})
    require(set(files) == {f'{STEM}.{ext}' for ext in ('stl', '3mf', 'glb')},
            'Incomplete or unexpected full-holder export set')
    for ext in ('stl', '3mf', 'glb'):
        path = DELIVERY/f'{STEM}.{ext}'
        row = files[path.name]
        check_hash(path, row.get('sha256'), ext+' export')
        require(path.stat().st_size == row['bytes'], f'{ext} export byte count differs')
        require(row.get('triangle_count') == report['source_triangle_count'],
                f'{ext} triangle count differs')
        if ext == 'stl':
            require(row.get('all_triangle_coordinates_exact') is True
                    and row.get('triangle_order_and_winding_preserved') is True
                    and row.get('canonical_triangle_sha256') == canonical
                    and row.get('units') == 'mm', 'STL coordinate validation incomplete')
        elif ext == '3mf':
            require(row.get('all_vertex_coordinates_exact_as_float64') is True
                    and row.get('triangle_indices_order_and_winding_preserved') is True
                    and row.get('canonical_triangle_sha256') == canonical
                    and row.get('units') == 'millimeter', '3MF coordinate validation incomplete')
        else:
            require(row.get('all_triangle_coordinates_verified') is True
                    and row.get('triangle_order_and_winding_preserved') is True
                    and row.get('maximum_coordinate_error_mm', float('inf'))
                        <= row.get('coordinate_error_tolerance_mm', -1)
                    and row.get('units') == 'metre'
                    and row.get('external_resources') is False,
                    'GLB coordinate validation incomplete')
    require((DELIVERY/f'{STEM}.3mf').stat().st_size <= UPLOAD_LIMIT_BYTES,
            'Direct 3MF exceeds the upload limit')


def validate_inputs():
    reports = {name: read_json(ROOT/'reports'/name) for name in REQUIRED_REPORTS}
    for name, report in reports.items():
        require(report.get('passed') is True, f'Incomplete validation: {name}')
    for n in (3, 4, 5):
        meta = read_json(CACHE/f'sphere_{n}layers.json')
        require(meta.get('version') == 5 and meta.get('layer_count') == n,
                f'Mixed or outdated sphere cache for {n} layers')
    for name in ('motion_validation.json', 'early_support_validation.json',
                 'strength_validation.json', 'full_height_validation.json'):
        validate_module_report(reports[name], name)
    check_hash(ROOT/'source'/'audit_full_height.py',
               reports['full_height_validation.json'].get('audit_source_sha256'),
               'whole-height audit source')
    # The assembly cache itself is stamped by build_holder.py with the
    # actual wall, sphere meshes and placement-source file it consumed.
    assembly_inputs = read_json(ROOT/'cache'/'assembly_inputs_supported.json')
    require(bool(assembly_inputs), 'Assembly input fingerprint is missing')
    for name, expected in assembly_inputs.items():
        check_hash(ROOT/name, expected, 'holder assembly input')
    preparation = read_json(ROOT/'reports'/'print_preparation.json')
    require(preparation.get('triangles') == reports['geometry_validation.json']['triangles']
            and preparation.get('watertight') is True
            and preparation.get('consistent_winding') is True,
            'Print preparation differs from the geometry audit')
    require(read_json(ROOT/'reports'/'support_projection_clearance.json').get('passed') is True,
            'Support projection validation incomplete')
    generation = read_json(ROOT/'reports'/'sphere_generation_checks.json')
    require(len(generation) == 3 and all(row.get('passed') is True for row in generation),
            'Sphere generation validation incomplete')
    strength = reports['strength_validation.json']
    require(strength.get('fingerprints_stable') is True, 'Strength inputs changed during audit')
    require(bool(strength.get('input_sha256')), 'Strength input fingerprints missing')
    for name, expected in strength['input_sha256'].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT), 'Strength input outside the packaged project')
        check_hash(path, expected, 'strength input')
    precision = reports['precision_repairs_scope.json']
    check_hash(ROOT/'reports'/'print_preparation.json',
               precision.get('preparation_report_sha256'), 'precision preparation')
    check_hash(ROOT/'cache'/'final.npz', precision.get('original_assembly_sha256'),
               'precision original assembly')
    access = reports['holder_service_access.json']
    check_hash(ROOT/'cache'/'wall_for_assembly.npz', access.get('wall_sha256'),
               'holder access wall')
    for placement in access['placements']:
        check_hash(CACHE/f"sphere_{placement['layers']}layers.json",
                   placement.get('module_metadata_sha256'), 'holder access metadata')
    coupon = reports['test_coupon_validation.json']
    require(coupon.get('inputs_unchanged_during_run') is True, 'Coupon inputs changed')
    require(bool(coupon.get('input_manifest')), 'Coupon input manifest missing')
    for item in coupon['input_manifest']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = ROOT/path
        check_hash(path, item['sha256'], 'coupon input')
    coupon_files = {Path(item['path']).name: item for item in coupon['exports']}
    require(set(coupon_files) == {COUPON_STEM+'.stl', COUPON_STEM+'.3mf'},
            'Coupon export set is incomplete')
    for name, item in coupon_files.items():
        path = ROOT/'models'/name
        check_hash(path, item['sha256'], 'coupon export')
        require(path.stat().st_size == item['bytes'], 'Coupon export byte count differs')
        validation = item['roundtrip_validation']
        coordinate_key = ('all_triangle_coordinates_exact' if name.endswith('.stl')
                          else 'all_vertex_coordinates_exact_as_float64')
        require(validation.get(coordinate_key) is True, 'Coupon coordinate validation incomplete')
    require(len({item['roundtrip_validation'].get('canonical_triangle_sha256')
                 for item in coupon_files.values()}) == 1,
            'Coupon STL and 3MF canonical triangle digests differ')
    export_path = DELIVERY/f'{STEM}_export_validation.json'
    validate_exports(read_json(export_path), reports['geometry_validation.json'])
    return export_path


def referenced_meshes(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from referenced_meshes(item)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_meshes(item)
    elif isinstance(value, str) and value.endswith('.npz'):
        require(Path(value).name == value, 'Sphere cache reference is not a basename')
        yield value


def collect_project(export_path):
    files = {}
    def add(path, relative):
        path, relative = Path(path), str(PurePosixPath(relative))
        require(path.is_file(), f'Missing project file: {path}')
        require(not PurePosixPath(relative).is_absolute()
                and '..' not in PurePosixPath(relative).parts, 'Unsafe ZIP member path')
        require(path.resolve().is_relative_to(ROOT), 'Project file escapes project directory')
        require(relative not in files, f'Duplicate ZIP member: {relative}')
        files[relative] = path
    add(ROOT/'README_JA.txt', 'README_JA.txt')
    add(ROOT/'PRINT_TEST_JA.txt', 'PRINT_TEST_JA.txt')
    add(export_path, 'reports/export_validation.json')
    for path in sorted((ROOT/'source').iterdir()):
        if path.suffix in ('.py', '.txt'):
            add(path, 'source/'+path.name)
    for name in REPORT_NAMES:
        add(ROOT/'reports'/name, 'reports/'+name)
    for n in (3, 4, 5):
        path = CACHE/f'sphere_{n}layers.json'
        meta = read_json(path)
        add(path, 'cache/spheres/'+path.name)
        for name in sorted(set(referenced_meshes(meta))):
            add(CACHE/name, 'cache/spheres/'+name)
    reference = ROOT/'reference'/'v3'
    require(reference.is_dir(), 'The packaged V3 comparison inputs are missing')
    for path in sorted(reference.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json', '.npz'):
            add(path, path.relative_to(ROOT).as_posix())
    for ext in ('stl', '3mf'):
        add(ROOT/'models'/f'{COUPON_STEM}.{ext}', f'models/{COUPON_STEM}.{ext}')
    # The main guide contains coupon instructions; include a separate guide
    # too if the owning workflow creates one with either supported name.
    for name in ('TEST_COUPON_JA.txt', 'README_TEST_COUPON_JA.txt'):
        if (ROOT/name).exists():
            add(ROOT/name, name)
    render_reports = {
        'final': ROOT/'previews'/'final'/'render_report_hero_top_views_detail.json',
        'mechanism': ROOT/'previews'/'mechanism'/'mechanism_render_report.json',
    }
    image_records = {}
    for kind, path in render_reports.items():
        data = read_json(path)
        if kind == 'final':
            check_hash(ROOT/'cache'/'print_ready_v5.npz', data.get('source_sha256'),
                       'final render source')
        else:
            check_hash(CACHE/'sphere_5layers.json', data.get('metadata_sha256'),
                       'mechanism render metadata')
            for row in data.get('source_parts', []):
                check_hash(ROOT/row['file'], row['sha256'], 'mechanism render part')
        image_records[kind] = {row['name']: row for row in data['images']}
        add(path, 'reports/'+path.name)
    for kind, original, label in IMAGE_MAP:
        target = DELIVERY/f'ivory_multisphere_pen_holder_v5_{label}.jpg'
        require(original in image_records[kind], f'Render report lacks {original}')
        check_hash(target, image_records[kind][original]['sha256'], 'copied preview image')
        add(target, 'images/'+target.name)
    coupon_render_path = ROOT/'reports'/'test_coupon_render.json'
    coupon_render = read_json(coupon_render_path)
    require(coupon_render.get('passed') is True, 'Coupon preview validation incomplete')
    check_hash(ROOT/'models'/f'{COUPON_STEM}.stl', coupon_render.get('source_sha256'),
               'coupon preview source')
    coupon_image = DELIVERY/f'{COUPON_STEM}.jpg'
    check_hash(coupon_image, coupon_render.get('output_sha256'), 'coupon preview image')
    add(coupon_render_path, 'reports/test_coupon_render.json')
    add(coupon_image, 'images/'+coupon_image.name)
    return files


class ArchiveTooLarge(RuntimeError):
    pass


def write_archive(name, files, snapshots):
    target = DELIVERY/name
    temporary = target.with_name(target.name+'.part')
    entries = {relative: snapshots[relative] for relative in sorted(files)}
    manifest = dict(schema_version=1, revision=REVISION, archive=name,
                    files=entries, manifest_self_hash='Excluded to avoid a circular digest')
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode('utf-8')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, path in sorted(files.items()):
                archive.write(path, relative)
            archive.writestr('archive_manifest.json', encoded)
        if temporary.stat().st_size > UPLOAD_LIMIT_BYTES:
            raise ArchiveTooLarge(f'{name}: {temporary.stat().st_size} bytes exceeds {UPLOAD_LIMIT_BYTES}')
        with zipfile.ZipFile(temporary) as archive:
            require(archive.testzip() is None, f'ZIP CRC failure: {name}')
            require(set(archive.namelist()) == set(files)|{'archive_manifest.json'},
                    f'ZIP member mismatch: {name}')
            for relative, expected in entries.items():
                h = hashlib.sha256()
                length = 0
                with archive.open(relative) as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b''):
                        h.update(block)
                        length += len(block)
                require(length == expected['bytes'] and h.hexdigest() == expected['sha256'],
                        f'ZIP member differs from input: {name}/{relative}')
            require(archive.read('archive_manifest.json') == encoded, 'ZIP manifest changed')
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return dict(path=str(target.resolve()), **record(target), members=len(files)+1,
                crc_passed=True, member_sha256_verified=True, upload_limit_passed=True)


def package_project(files, snapshots):
    """Keep the project together unless actual compressed size requires parts."""
    try:
        return [write_archive('ivory_v5_01_project.zip', files, snapshots)]
    except ArchiveTooLarge:
        pass
    images = {name: path for name, path in files.items() if name.startswith('images/')}
    core = {name: path for name, path in files.items() if name not in images}
    image_archive = write_archive('ivory_v5_04_images.zip', images, snapshots)
    try:
        return [write_archive('ivory_v5_01_project.zip', core, snapshots), image_archive]
    except ArchiveTooLarge:
        pass
    # Each mechanism and the comparison inputs remain ordinary independent
    # archives. No binary segmentation and no geometric simplification occur.
    groups = []
    taken = set()
    for n in (3, 4, 5):
        meta_name = f'cache/spheres/sphere_{n}layers.json'
        meta = read_json(files[meta_name])
        names = {meta_name}|{'cache/spheres/'+v for v in referenced_meshes(meta)}
        groups.append((f'ivory_v5_cache_{n}layers.zip', {name: files[name] for name in names}))
        taken.update(names)
    reference = {name: path for name, path in core.items() if name.startswith('reference/')}
    groups.append(('ivory_v5_reference_v3.zip', reference))
    taken.update(reference)
    remaining = {name: path for name, path in core.items() if name not in taken}
    result = [write_archive('ivory_v5_01_project.zip', remaining, snapshots), image_archive]
    result.extend(write_archive(name, members, snapshots) for name, members in groups)
    return result


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    export_path = validate_inputs()
    project = collect_project(export_path)
    full_models = {f'models/{STEM}.{ext}': DELIVERY/f'{STEM}.{ext}'
                   for ext in ('3mf', 'stl', 'glb')}
    all_files = {**project, **full_models}
    snapshots = {name: record(path) for name, path in sorted(all_files.items())}
    manifest = dict(schema_version=1, revision=REVISION, units='mm',
                    upload_limit_bytes=UPLOAD_LIMIT_BYTES, files=snapshots,
                    restore='Extract every supplied ordinary project/cache/reference/images ZIP into one directory. The complete holder 3MF is supplied separately; STL and GLB are in their own ZIPs.',
                    physical_print_tested=False)
    manifest_path = ROOT/'reports'/'delivery_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    project['delivery_manifest.json'] = manifest_path
    snapshots['delivery_manifest.json'] = record(manifest_path)
    archives = package_project(project, snapshots)
    for number, ext in ((2, 'stl'), (3, 'glb')):
        member = f'models/{STEM}.{ext}'
        archives.append(write_archive(f'ivory_v5_{number:02d}_{ext}.zip',
                                      {member: full_models[member]}, snapshots))
    # Rehash once after the entire run so concurrent regeneration cannot
    # produce a project assembled from incompatible snapshots.
    for name, path in all_files.items():
        require(record(path) == snapshots[name], f'Input changed while packaging: {name}')
    direct = [DELIVERY/f'{STEM}.3mf', ROOT/'models'/f'{COUPON_STEM}.3mf',
              ROOT/'models'/f'{COUPON_STEM}.stl', ROOT/'README_JA.txt', ROOT/'PRINT_TEST_JA.txt']
    direct.extend(path for name, path in project.items() if name.startswith('images/'))
    direct_files = []
    for path in direct:
        item = dict(path=str(path.resolve()), **record(path))
        require(item['bytes'] <= UPLOAD_LIMIT_BYTES, f'Direct file exceeds upload limit: {path}')
        direct_files.append(item)
    inventory = dict(schema_version=1, revision=REVISION, passed=True,
                     upload_limit_bytes=UPLOAD_LIMIT_BYTES, archives=archives,
                     direct_files=direct_files, inputs_unchanged_during_packaging=True,
                     manifest_path=str(manifest_path.resolve()),
                     archive_format='Ordinary self-contained ZIP archives; not split binary volumes')
    inventory_path = DELIVERY/'ivory_v5_delivery_inventory.json'
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, inventory_path=str(inventory_path.resolve()),
                         archives=archives, direct_file_count=len(direct_files)), indent=2))


if __name__ == '__main__':
    main()
