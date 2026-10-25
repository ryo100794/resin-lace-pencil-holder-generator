#!/usr/bin/env python3
"""Freeze and archive the immutable 1037/1053 removal-analysis checkpoint.

No scene test, source regeneration, printed-input edit or active checkpoint
read is performed. All expected proof/input/output hashes must match before
any archive is considered complete. Current source snapshots are separate.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PROOF = 'reports/external_removal_endpoint_partial_d2e85a5d25ea33e2.json'
PROOF_SHA = 'd2e85a5d25ea33e2419f7999e293efe8ddc0f2bfc8409f4a6e3f7efa37e19e76'
NAME = 'V6_WORKING_removal_checkpoint_07.zip'
LIMIT = 100 * 1024 * 1024
BLOCK = 1024 * 1024
ACTIVE_NAMES = {'checkpoint.json', 'v6_external_removal_validation.json'}


def hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def validate_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unexpected archive path')
    resolved = (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    if not resolved.is_file() or path.name in ACTIVE_NAMES:
        raise ValueError('Missing or actively rewritten source is not allowed: ' + name)
    return resolved


def main():
    proof_data = (ROOT / PROOF).read_bytes()
    if hashlib.sha256(proof_data).hexdigest() != PROOF_SHA:
        raise ValueError('Immutable partial proof hash mismatch')
    proof = json.loads(proof_data)
    if len(proof['plans']) != 1037 or len(proof['remaining_failures']) != 16:
        raise ValueError('Unexpected checkpoint counts')
    expected, roles = {}, {}

    def bind(path, digest, role):
        if path in expected and expected[path] != digest:
            raise ValueError('Conflicting immutable hash: ' + path)
        expected[path] = digest
        roles.setdefault(path, set()).add(role)

    bind(PROOF, PROOF_SHA, 'immutable_completed_partial_proof')
    for entry in proof['input_files']:
        bind(entry['path'], entry['sha256'], 'proof_bound_input')
    for row in proof['plans']:
        for fragment in row['fragments']:
            bind(fragment['mesh_file'], fragment['sha256'], 'accepted_operation_saved_fragment')
        if row.get('actual_body_kerf'):
            entry = row['actual_body_kerf']
            bind(entry['path'], entry['sha256'], 'accepted_operation_saved_body_kerf')
    staging_parent = ROOT / 'cache/working_checkpoint07_staging'
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='snapshot_', dir=staging_parent))
    records = []

    def freeze(source, archive_path, expected_hash, member_roles, source_relative):
        destination = stage / archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            for block in iter(lambda: incoming.read(BLOCK), b''):
                digest.update(block)
                count += len(block)
                outgoing.write(block)
        found = digest.hexdigest()
        if expected_hash is not None and found != expected_hash:
            raise ValueError('Source bytes drifted from immutable proof: ' + source_relative)
        disk_size, disk_hash = hash_file(destination)
        if (disk_size, disk_hash) != (count, found):
            raise ValueError('Frozen copy failed disk readback: ' + archive_path)
        records.append(dict(path=archive_path, source_relative_path=source_relative,
            bytes=count, sha256=found, roles=sorted(member_roles),
            bound_to_partial_proof=expected_hash is not None))

    for number, (path, digest) in enumerate(sorted(expected.items()), 1):
        freeze(validate_relative(path), path, digest, roles[path], path)
        if number % 2500 == 0:
            print(json.dumps(dict(stage='freezing_bound_files', done=number, total=len(expected))), flush=True)
    # These bytes are intentionally labeled separately. Unbound current code
    # is not retroactively claimed as a source of the older partial proof.
    sources = sorted(set((ROOT / 'source').glob('*.py'))
                     | set((ROOT / 'source').glob('requirements*.txt')))
    for source in sources:
        relative = str(source.relative_to(ROOT))
        freeze(source, 'current_source_snapshot/' + relative, None,
               {'current_source_snapshot_not_retroactively_proof_bound'}, relative)
    residual = [row['id'] for row in proof['remaining_failures']]
    status = (
        'V6 WORKING REMOVAL-ANALYSIS CHECKPOINT 07\n'
        'NOT A FINAL PRINT RELEASE / 最終印刷リリースではありません\n\n'
        'This is a durable intermediate removal-analysis checkpoint.\n'
        'Verified accepted operations in the enclosed immutable partial proof: 1037 / 1053.\n'
        'Unresolved operations at this checkpoint: 16.\n'
        'The print geometry was already frozen separately. Creating this archive made no print-input changes.\n'
        'No new scene, cutting, physical tool-access, print-strength or route tests were run for this archive.\n\n'
        'このZIPは取り外し検討の途中保存です。検証済み1,037件、未解決16件の時点を保存しています。\n'
        '造形形状は別途固定済みです。この保存処理では造形用入力を変更していません。\n'
        '現在の探索が進んでも、このZIPの検証結果や未解決件数は書き換わりません。\n\n'
        f'Immutable proof: {PROOF}\nSHA256: {PROOF_SHA}\n\n'
        'Contents: every input byte-bound by that proof, every referenced accepted fragment and actual body kerf,\n'
        'and a separately labeled current_source_snapshot/source/ directory of current Python sources and requirements.\n'
        'The current source snapshot includes newer controllers but is not asserted to belong to the older proof.\n'
        'Restore proof-bound files at their relative paths to inspect or reproduce the saved analysis.\n'
        'Do not treat the source snapshot as a new passed proof or a final release authorization.\n\n'
        'Unresolved IDs:\n' + '\n'.join(residual) + '\n\n'
        'CHECKPOINT07_MEMBER_HASHES.json lists every payload/readme member with exact bytes and SHA256.\n'
        'The manifest excludes its own hash to avoid a circular definition; the external validation/Drive receipt\n'
        'binds the manifest and ZIP hashes. Every ZIP member is read back for CRC and SHA256 verification.\n')
    readme = stage / 'README_WORKING_CHECKPOINT07.txt'
    readme.write_text(status, encoding='utf-8')
    readme_size, readme_hash = hash_file(readme)
    records.append(dict(path=readme.name, source_relative_path=None, bytes=readme_size,
                        sha256=readme_hash, roles=['checkpoint_status'], bound_to_partial_proof=False))
    records.sort(key=lambda row: row['path'])
    manifest = dict(schema='working_removal_checkpoint07_member_inventory_v1',
        status='WORKING removal-analysis checkpoint; NOT final print release',
        immutable_proof_path=PROOF, immutable_proof_sha256=PROOF_SHA,
        accepted_operations=1037, unresolved_operations=16, unresolved_ids=residual,
        original_bound_input_count=len(proof['input_files']),
        proof_and_output_payload_count=len(expected), current_source_snapshot_count=len(sources),
        print_geometry_modified=False, actively_rewritten_checkpoint_files_included=False,
        members=records, self_hash_excluded_to_avoid_circular_manifest=True)
    manifest_name = 'CHECKPOINT07_MEMBER_HASHES.json'

    def write_zip(path, selected, archive_names):
        own_manifest = dict(manifest, archive_set=archive_names,
            this_archive=path.name, member_paths_in_this_archive=[r['path'] for r in selected])
        manifest_bytes = encode(own_manifest)
        with zipfile.ZipFile(path, 'w', allowZip64=True) as archive:
            for row in selected:
                compression = zipfile.ZIP_STORED if row['path'].endswith('.npz') else zipfile.ZIP_DEFLATED
                archive.write(stage / row['path'], row['path'], compress_type=compression, compresslevel=6)
            archive.writestr(manifest_name, manifest_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        return hashlib.sha256(manifest_bytes).hexdigest()

    deliverables = ROOT / 'deliverables'
    deliverables.mkdir(exist_ok=True)
    first_path = deliverables / NAME
    if first_path.exists():
        raise ValueError('Checkpoint output already exists; refusing to overwrite')
    manifest_hash = write_zip(first_path, records, [NAME])
    archives = [(first_path, records, manifest_hash)]
    if first_path.stat().st_size >= LIMIT:
        # Both ordinary ZIP parts are complete, with explicit full inventory.
        with zipfile.ZipFile(first_path) as trial:
            sizes = {info.filename: info.compress_size for info in trial.infolist()}
        names = ['V6_WORKING_removal_checkpoint_07_part1.zip', 'V6_WORKING_removal_checkpoint_07_part2.zip']
        groups, totals = [[], []], [0, 0]
        for row in sorted(records, key=lambda r: sizes.get(r['path'], r['bytes']), reverse=True):
            if row['path'] == readme.name:
                continue
            chosen = 0 if totals[0] <= totals[1] else 1
            groups[chosen].append(row)
            totals[chosen] += sizes.get(row['path'], row['bytes'])
        archives = []
        for name, group in zip(names, groups):
            group.append(next(r for r in records if r['path'] == readme.name))
            group.sort(key=lambda r: r['path'])
            path = deliverables / name
            if path.exists():
                raise ValueError('Split checkpoint output already exists')
            mh = write_zip(path, group, names)
            archives.append((path, group, mh))
        first_path.unlink()
    validated = []
    for path, selected, mh in archives:
        if path.stat().st_size >= LIMIT:
            raise ValueError('An ordinary checkpoint ZIP exceeds the requested 100 MiB cap')
        by_name = {row['path']: row for row in selected}
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist()) != len(by_name) + 1 or set(archive.namelist()) != set(by_name) | {manifest_name}:
                raise ValueError('Archive member inventory mismatch')
            for name in archive.namelist():
                digest = hashlib.sha256()
                count = 0
                # Reading to EOF validates each stored CRC in zipfile too.
                with archive.open(name) as incoming:
                    for block in iter(lambda: incoming.read(BLOCK), b''):
                        count += len(block)
                        digest.update(block)
                expected_digest = mh if name == manifest_name else by_name[name]['sha256']
                if digest.hexdigest() != expected_digest:
                    raise ValueError('ZIP member hash mismatch: ' + name)
                if name != manifest_name and count != by_name[name]['bytes']:
                    raise ValueError('ZIP member size mismatch: ' + name)
        count, digest = hash_file(path)
        validated.append(dict(path=str(path.relative_to(ROOT)), name=path.name, bytes=count,
            sha256=digest, member_count=len(selected) + 1, member_manifest_sha256=mh,
            every_member_crc_verified=True, every_payload_member_sha256_and_size_verified=True))
    report = dict(passed=True, schema='working_removal_checkpoint07_validation_v1',
        status=manifest['status'], checked_at_utc=datetime.now(timezone.utc).isoformat(),
        immutable_proof_path=PROOF, immutable_proof_sha256=PROOF_SHA,
        accepted_operations=1037, unresolved_operations=16,
        proof_bound_input_count=len(proof['input_files']), frozen_proof_and_output_files=len(expected),
        current_source_snapshot_count=len(sources), all_expected_source_hashes_matched=True,
        frozen_copies_read_back_before_zip=True, original_print_inputs_modified=False,
        active_checkpoint_files_read=False, no_new_geometry_tests=True, archives=validated)
    output = ROOT / 'reports/working_checkpoint07_local_validation.json'
    output.write_bytes(encode(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
