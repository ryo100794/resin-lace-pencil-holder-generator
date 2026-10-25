"""Reconstruct the five unchanged V6 binaries from their stored byte parts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BLOCK_BYTES = 1024 * 1024


def local_path(value: str) -> Path:
    rel = PurePosixPath(value)
    if not value or rel.is_absolute() or '..' in rel.parts or '\\' in value:
        raise ValueError(f'Invalid relative path: {value!r}')
    path = ROOT.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Path escapes this folder: {value!r}')
    return path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(BLOCK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict) -> None:
    local_path(record['path'])
    if not isinstance(record['bytes'], int) or record['bytes'] <= 0:
        raise ValueError('Invalid original byte count')
    if not re.fullmatch('[0-9a-f]{64}', record['sha256']):
        raise ValueError('Invalid original SHA-256')
    if not record['parts'] or sum(p['bytes'] for p in record['parts']) != record['bytes']:
        raise ValueError(f'Part sizes do not sum to original size: {record["path"]}')
    names = [p['path'] for p in record['parts']]
    if len(set(names)) != len(names) or record['path'] in names:
        raise ValueError(f'Duplicate or circular part path: {record["path"]}')
    for part in record['parts']:
        local_path(part['path'])
        if not isinstance(part['bytes'], int) or not 0 < part['bytes'] <= 32 * 1024 * 1024:
            raise ValueError(f'Invalid part size: {part["path"]}')
        if not re.fullmatch('[0-9a-f]{64}', part['sha256']):
            raise ValueError(f'Invalid part SHA-256: {part["path"]}')


def stream_parts(record: dict, output=None) -> dict:
    whole = hashlib.sha256()
    total = 0
    checked = []
    for part in record['parts']:
        path = local_path(part['path'])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or nonregular part: {part["path"]}')
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(BLOCK_BYTES), b''):
                count += len(chunk)
                digest.update(chunk)
                whole.update(chunk)
                if output is not None:
                    output.write(chunk)
        actual = digest.hexdigest()
        if count != part['bytes'] or actual != part['sha256']:
            raise ValueError(f'Part size or SHA-256 mismatch: {part["path"]}')
        total += count
        checked.append({'path': part['path'], 'bytes': count, 'sha256': actual})
    actual = whole.hexdigest()
    if total != record['bytes'] or actual != record['sha256']:
        raise ValueError(f'Reconstructed size or SHA-256 mismatch: {record["path"]}')
    return {'path': record['path'], 'bytes': total, 'sha256': actual,
            'parts': checked, 'passed': True, 'materialized': False}


def process(record: dict, verify_only: bool) -> dict:
    validate_record(record)
    if verify_only:
        return stream_parts(record)

    target = local_path(record['path'])
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.stat().st_size == record['bytes'] and file_hash(target) == record['sha256']:
            return {'path': record['path'], 'bytes': record['bytes'], 'sha256': record['sha256'],
                    'passed': True, 'materialized': False, 'skipped_matching_existing': True}
        raise FileExistsError(f'Existing file differs; move it aside before restoring: {record["path"]}')

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent,
                                         prefix='.' + target.name + '.restore-',
                                         delete=False) as output:
            temp_path = Path(output.name)
            result = stream_parts(record, output)
            output.flush()
            os.fsync(output.fileno())
        # A same-directory hard link publishes the verified bytes atomically
        # without replacing any target created after the initial check.
        # Unsupported filesystems fail safely and the temporary file is removed.
        try:
            os.link(temp_path, target)
        except FileExistsError:
            if target.is_file() and not target.is_symlink() and target.stat().st_size == record['bytes'] and file_hash(target) == record['sha256']:
                result['skipped_matching_existing'] = True
                return result
            raise FileExistsError(f'Output appeared during restoration; left unchanged: {record["path"]}')
        temp_path.unlink()
        temp_path = None
        result['materialized'] = True
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true',
                        help='Stream and verify all parts and complete-file hashes without writing reconstructed binaries.')
    parser.add_argument('--json', action='store_true', help='Print a JSON verification result.')
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / 'BINARY_PARTS.json').read_text(encoding='utf-8'))
        if manifest['schema'] != 'penholder_v6_binary_parts_v1':
            raise ValueError('Unsupported BINARY_PARTS.json schema')
        names = [r['path'] for r in manifest['files']]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate original output paths')
        results = [process(record, args.verify) for record in manifest['files']]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    report = {'schema': 'penholder_v6_binary_parts_validation_v1', 'passed': True,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'verify_only': args.verify, 'geometry_tests_performed': False,
              'original_files': len(results),
              'stored_parts': sum(len(r['parts']) for r in manifest['files']),
              'scope': 'Ordered parts match their recorded hashes and reproduce exact original bytes. No geometry or print tests.',
              'files': results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in results:
            state = 'VERIFIED' if args.verify else ('SKIPPED matching existing' if row.get('skipped_matching_existing') else 'RESTORED')
            print(f'{state}: {row["path"]} ({row["bytes"]} bytes)')
        print(f'PASS: {len(results)} original files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
