from pathlib import Path, PurePosixPath
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
failures = []
for row in manifest["files"]:
    rel = PurePosixPath(row["path"])
    path = root.joinpath(*rel.parts)
    if rel.is_absolute() or ".." in rel.parts or not path.resolve().is_relative_to(root):
        failures.append((row["path"], "unsafe path"))
        continue
    if not path.is_file() or path.is_symlink():
        failures.append((row["path"], "missing or nonregular file"))
        continue
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(data)
            git_sha1.update(data)
    if size != row["bytes"] or sha256.hexdigest() != row["sha256"] or git_sha1.hexdigest() != row["git_blob_sha1"]:
        failures.append((row["path"], "content mismatch"))
if failures:
    for path, reason in failures:
        print(f"FAIL {path}: {reason}")
    sys.exit(1)
print(f"PASS: {len(manifest['files'])} stored files match size, SHA-256 and Git blob SHA-1.")
print("Run python restore_parts.py --verify to check reconstructed-original hashes without writing originals.")
