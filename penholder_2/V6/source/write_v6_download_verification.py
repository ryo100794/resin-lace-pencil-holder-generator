#!/usr/bin/env python3
"""Consolidate V6 delivery integrity AFTER packaging/upload/raw readbacks.

This is a filesystem-only postupload report writer. It does not perform
geometry tests, call Drive, upload files or change routing inputs. Every
publish file is freshly hashed locally and matched to the package report
and canonical Drive upload metadata. Actual remote-byte verification is
claimed only for matching, successful raw-download evidence.

Required inputs:
  reports/v6_package_validation.json
  reports/drive_v6_final_uploads.json
  reports/drive_v6_preview_readback.json
  reports/drive_v6_main3mf_readback.json
  reports/drive_v6_archive_readback.json

The future archive readback uses the preview report's records schema; the
same records list may also include the four guide/index/checksum files:
  {"passed": true,
   "fetch_options": {"download_raw_file": true, "include_base64": false},
   "checked_at_utc": "<ISO timestamp>",
   "records": [{"passed": true, "id": "<Drive ID>", "name": "<name>",
     "url": "<stable Drive URL>", "parent_id": "<V6 folder ID>",
     "expected_bytes": 123, "expected_sha256": "<SHA256>",
     "http_status": 200, "downloaded_bytes": 123,
     "downloaded_sha256": "<SHA256>", "disk_readback_bytes": 123,
     "disk_readback_sha256": "<SHA256>"}]}
The existing main-3MF report is accepted in its equivalent flat schema.
If evidence also records fetch_file_size_bytes, source_local_bytes/hash,
or bytes_match/sha256_match/disk_readback_verified, those must agree too.

All nine large/visual primary files require actual raw evidence: 3MF,
four previews, STL ZIP, project ZIP, support-assets ZIP and repair-history
ZIP. Guides/index/checksum list receive metadata_only scope unless actual
raw evidence is supplied. Missing or mismatched evidence aborts writing.

Output: deliverables/DOWNLOAD_VERIFICATION.json. It is intentionally a
standalone postupload file, never a member of its own package publish list.
Only stable Drive URLs, public filenames and project-relative provenance
paths are emitted; private download references/local scratch paths are not.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'deliverables'
OUTPUT_NAME = 'DOWNLOAD_VERIFICATION.json'
PACKAGE_REPORT = 'reports/v6_package_validation.json'
UPLOAD_REPORT = 'reports/drive_v6_final_uploads.json'
RAW_REPORTS = (
    'reports/drive_v6_preview_readback.json',
    'reports/drive_v6_main3mf_readback.json',
    'reports/drive_v6_archive_readback.json',
)
REQUIRED_RAW = frozenset({
    'ivory_penholder_v6_custom_supported.3mf',
    'ivory_v6_support_overview.jpg',
    'ivory_v6_sphere_support_detail.jpg',
    'ivory_v6_wall_support_detail.jpg',
    'ivory_v6_base_support_detail.jpg',
    'ivory_v6_stl.zip',
    'ivory_v6_project.zip',
    'ivory_v6_support_assets.zip',
    'ivory_v6_repair_history.zip',
})
SHA = re.compile(r'^[0-9a-f]{64}$')
DRIVE_ID = re.compile(r'^[A-Za-z0-9_-]+$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, label):
    require(not isinstance(value, bool), 'Boolean used as a byte count: '+label)
    if isinstance(value, int):
        result = value
    else:
        require(isinstance(value, str) and value.isdigit(), 'Invalid integer: '+label)
        result = int(value)
    require(result >= 0, 'Negative integer: '+label)
    return result


def checksum(value, label):
    require(isinstance(value, str) and SHA.fullmatch(value) is not None,
            'Invalid SHA256: '+label)
    return value


def filename(value):
    require(isinstance(value, str) and value not in ('', '.', '..')
            and '/' not in value and '\\' not in value
            and not any(ord(c) < 32 for c in value), 'Invalid public filename')
    return value


def stable_drive_url(value, file_id, label):
    """Allow observed canonical Drive file URLs, never bearer download URLs."""
    require(isinstance(file_id, str) and DRIVE_ID.fullmatch(file_id) is not None,
            'Invalid Drive ID: '+label)
    require(isinstance(value, str), 'Missing stable Drive URL: '+label)
    parsed = urlsplit(value)
    require(parsed.scheme == 'https' and parsed.netloc == 'drive.google.com'
            and not parsed.fragment, 'Noncanonical Drive URL: '+label)
    query = parse_qs(parsed.query, keep_blank_values=True)
    require(set(query) <= {'usp'} and all(len(v) == 1 for v in query.values()),
            'Unexpected query on stable Drive URL: '+label)
    require(all(value in {'drivesdk', 'sharing', 'drive_link'}
                for values in query.values() for value in values),
            'Unexpected query value on stable Drive URL: '+label)
    require(parsed.path in ('/file/d/'+file_id+'/view', '/file/d/'+file_id),
            'Stable URL/Drive ID mismatch: '+label)
    return value


def evidence_time(value):
    require(isinstance(value, str), 'Missing raw readback timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Raw readback timestamp lacks a timezone')
    return parsed.isoformat()


def file_hash(path):
    """Bounded-memory actual file read, with concurrent-change detection."""
    before = path.stat()
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
            size += len(block)
    after = path.stat()
    fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == fingerprint,
            'Publish file changed while hashing: '+path.name)
    require(size == after.st_size, 'Publish file read length differs: '+path.name)
    return size, h.hexdigest(), fingerprint


def read_report(relative, provenance):
    path = ROOT/relative
    require(path.is_file(), 'Missing required evidence report: '+relative)
    raw = path.read_bytes()
    data = json.loads(raw)
    require(isinstance(data, dict), 'Evidence report is not an object: '+relative)
    entry = dict(path=relative, bytes=len(raw), sha256=digest(raw))
    provenance[relative] = entry
    return data, entry


def read_raw_evidence(provenance):
    by_id = {}
    for relative in RAW_REPORTS:
        report, report_entry = read_report(relative, provenance)
        require(report.get('passed') is True, 'Raw readback report has not passed: '+relative)
        options = report.get('fetch_options', {})
        require(options.get('download_raw_file') is True and options.get('include_base64') is False,
                'Raw readback did not use the required complete-file path: '+relative)
        require(report.get('signed_download_references_persisted', False) is False,
                'Raw evidence claims private download references were persisted: '+relative)
        if 'records' in report:
            rows = report['records']
            require(isinstance(rows, list) and rows, 'Empty raw evidence record list: '+relative)
            if 'receipt_records_unchanged' in report:
                require(report['receipt_records_unchanged'] is True,
                        'Readback receipt changed during verification: '+relative)
        else:
            rows = [report]
        timestamp = evidence_time(report.get('checked_at_utc'))
        for row in rows:
            require(isinstance(row, dict) and row.get('passed') is True,
                    'Unsuccessful raw readback row: '+relative)
            fid = row.get('id')
            require(isinstance(fid, str) and DRIVE_ID.fullmatch(fid) is not None,
                    'Invalid raw-evidence Drive ID: '+relative)
            by_id.setdefault(fid, []).append((row, report_entry, timestamp))
    return by_id


def verify_raw(row, report_entry, timestamp, published, upload, folder_id):
    name, size, sha256 = published['name'], published['bytes'], published['sha256']
    require(filename(row.get('name')) == name, 'Raw readback filename differs: '+name)
    require(row.get('id') == upload['id'] and row.get('parent_id') == folder_id,
            'Raw readback file/folder differs: '+name)
    stable_drive_url(row.get('url'), upload['id'], name)
    require(integer(row.get('http_status'), name+' HTTP status') == 200,
            'Raw readback HTTP status is not200: '+name)
    for field in ('expected_bytes', 'downloaded_bytes', 'disk_readback_bytes'):
        require(integer(row.get(field), name+' '+field) == size,
                'Raw readback byte count differs: '+name+' '+field)
    for field in ('expected_sha256', 'downloaded_sha256', 'disk_readback_sha256'):
        require(checksum(row.get(field), name+' '+field) == sha256,
                'Raw readback SHA256 differs: '+name+' '+field)
    for field in ('fetch_file_size_bytes', 'source_local_bytes'):
        if field in row:
            require(integer(row[field], name+' '+field) == size,
                    'Raw evidence local/fetch byte count differs: '+name)
    if 'source_local_sha256' in row:
        require(checksum(row['source_local_sha256'], name+' source_local_sha256') == sha256,
                'Raw evidence local-source SHA256 differs: '+name)
    for field in ('bytes_match', 'sha256_match', 'disk_readback_verified', 'receipt_record_unchanged'):
        if field in row:
            require(row[field] is True, 'Raw readback predicate is false: '+name+' '+field)
    # Explicit whitelist: never copy local_readback_path, download references,
    # source-local absolute paths or arbitrary descriptive input fields.
    return dict(report=report_entry['path'], report_sha256=report_entry['sha256'],
                checked_at_utc=timestamp, http_status=200,
                actual_download_bytes=size, actual_download_sha256=sha256,
                downloaded_file_reopened_and_hashed=True,
                disk_readback_bytes=size, disk_readback_sha256=sha256)


def build_report():
    provenance = {}
    package, _ = read_report(PACKAGE_REPORT, provenance)
    uploads, _ = read_report(UPLOAD_REPORT, provenance)
    require(package.get('passed') is True, 'Final local package validation has not passed')
    require(package.get('version') == 6 and uploads.get('version') == 6,
            'Package/upload version is not6')
    folder_id = uploads.get('folder_id')
    require(isinstance(folder_id, str) and DRIVE_ID.fullmatch(folder_id) is not None,
            'Missing canonical V6 Drive folder ID')
    published_rows = package.get('publish_files')
    require(isinstance(published_rows, list) and published_rows, 'Empty package publish list')
    published = {}
    for row in published_rows:
        name = filename(row.get('name'))
        require(name not in published, 'Duplicate package filename: '+name)
        require(name != OUTPUT_NAME,
                'Standalone DOWNLOAD_VERIFICATION.json must not be in its own package publish list')
        published[name] = dict(name=name, bytes=integer(row.get('bytes'), name),
                               sha256=checksum(row.get('sha256'), name))
    require(REQUIRED_RAW <= set(published), 'Package is missing one of the nine required raw-readback files')

    upload_rows = uploads.get('records')
    require(isinstance(upload_rows, list), 'Upload receipt has no records')
    by_name = {}
    seen_ids = set()
    for row in upload_rows:
        name = filename(row.get('name'))
        require(name not in by_name, 'Ambiguous duplicate filename in canonical upload receipt: '+name)
        require(row.get('id') not in seen_ids, 'Duplicate Drive ID in canonical upload receipt')
        by_name[name] = row
        seen_ids.add(row.get('id'))
    raw_by_id = read_raw_evidence(provenance)
    results = []
    local_fingerprints = {}
    for name, entry in published.items():
        require(name in by_name, 'Publish file has not been uploaded: '+name)
        row = by_name[name]
        file_id = row.get('id')
        url = stable_drive_url(row.get('url'), file_id, name)
        require(row.get('parent_id') == folder_id, 'Upload receipt parent differs: '+name)
        require(row.get('metadata_parent_and_size_verified') is True,
                'Upload metadata readback is not verified: '+name)
        require(integer(row.get('local_bytes'), name+' upload bytes') == entry['bytes']
                and checksum(row.get('local_sha256'), name+' upload SHA256') == entry['sha256'],
                'Package and upload receipt bytes differ: '+name)
        metadata = row.get('metadata', {})
        require(metadata.get('id') == file_id and metadata.get('title', metadata.get('name')) == name,
                'Drive metadata identity/name differs: '+name)
        require(metadata.get('file_or_folder') == 'file', 'Drive metadata is not a file: '+name)
        require(folder_id in metadata.get('parent_ids', []), 'Drive metadata parent differs: '+name)
        require(integer(metadata.get('size'), name+' metadata size') == entry['bytes'],
                'Drive metadata size differs: '+name)
        stable_drive_url(metadata.get('url'), file_id, name+' metadata')
        mime = metadata.get('mime_type')
        require(isinstance(mime, str) and re.fullmatch(r'[A-Za-z0-9.+_-]+/[A-Za-z0-9.+_-]+', mime),
                'Invalid Drive metadata MIME type: '+name)
        path = OUT/name
        require(path.is_file() and path.resolve().parent == OUT.resolve(),
                'Missing or indirect publish file: '+name)
        actual_size, actual_sha, fingerprint = file_hash(path)
        require((actual_size, actual_sha) == (entry['bytes'], entry['sha256']),
                'Local publish file changed after packaging: '+name)
        local_fingerprints[name] = fingerprint
        evidence = [verify_raw(raw, source, timestamp, entry, row, folder_id)
                    for raw, source, timestamp in raw_by_id.get(file_id, [])]
        require(evidence or name not in REQUIRED_RAW, 'Required actual raw readback is missing: '+name)
        raw_verified = bool(evidence)
        results.append(dict(
            name=name, bytes=actual_size, sha256=actual_sha,
            drive_id=file_id, drive_url=url, parent_folder_id=folder_id,
            mime_type=mime, package_and_local_bytes_verified=True,
            drive_metadata_identity_parent_and_size_verified=True,
            verification_scope='raw_verified' if raw_verified else 'metadata_only',
            actual_raw_download_verified=raw_verified,
            raw_download_evidence=evidence,
            verification_note=(
                'Actual authenticated raw download and reopened downloaded-file bytes match this SHA256 and size.'
                if raw_verified else
                'Local package bytes and Drive metadata identity, parent and size match. Actual remote bytes have not been independently read back.')))

    # Binding is to the exact input report bytes consumed above. In particular,
    # a readback's older receipt snapshot need not equal the later final
    # receipt: every relevant file ID/name/hash/size was matched individually.
    for relative, entry in provenance.items():
        require(digest((ROOT/relative).read_bytes()) == entry['sha256'],
                'Evidence report changed during consolidation: '+relative)
    for name, fingerprint in local_fingerprints.items():
        stat = (OUT/name).stat()
        require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) == fingerprint,
                'Publish file changed during consolidation: '+name)
    source = Path(__file__).resolve()
    raw_count = sum(row['actual_raw_download_verified'] for row in results)
    return dict(
        schema='v6_postupload_download_verification_v1', passed=True, version=6,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Postupload delivery integrity. Metadata-only checks and actual raw-byte readbacks are distinguished per file; this script performs no geometry or removal-route tests.',
        parent_folder_id=folder_id,
        publish_file_count=len(results), raw_verified_count=raw_count,
        metadata_only_count=len(results)-raw_count,
        required_raw_file_names=sorted(REQUIRED_RAW),
        all_required_raw_readbacks_verified=True,
        all_publish_files_match_local_package_and_drive_metadata=True,
        files=results,
        provenance_reports=[provenance[key] for key in sorted(provenance)],
        writer_source=dict(path=str(source.relative_to(ROOT)), bytes=source.stat().st_size,
                           sha256=digest(source.read_bytes())),
        network_downloads_performed_by_this_consolidation=False,
        raw_download_claim_basis='The byte-bound readback reports listed for each file, matched again to the final canonical upload receipt and freshly hashed local file.',
        geometry_or_support_removal_validation_performed=False,
        private_download_references_or_scratch_paths_included=False,
        standalone_report_policy='Generated after uploads. Intentionally excluded from the package publish manifest and its own checksum inventory to avoid a self-reference cycle.')


def main():
    report = build_report()
    target = OUT/OUTPUT_NAME
    temporary = target.with_suffix('.json.part')
    serialized = json.dumps(report, indent=2, ensure_ascii=False)+'\n'
    require('X-Amz-' not in serialized and 'download_url' not in serialized,
            'Unexpected private download-reference field in public output')
    temporary.write_text(serialized)
    temporary.replace(target)
    print(json.dumps(dict(passed=True, output='deliverables/'+OUTPUT_NAME,
        publish_file_count=report['publish_file_count'], raw_verified_count=report['raw_verified_count'],
        metadata_only_count=report['metadata_only_count'], sha256=digest(target.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
