#!/usr/bin/env python3
"""Restore only the regenerable full-holder archive from frozen final parts."""
from pathlib import Path
import hashlib,io,json,os,time,zipfile
import numpy as np
import custom_holder_supports as H

ROOT=H.ROOT;OUT=H.OUT
EXPECTED='ed20e263bd7ff2e6cf2badcdeb9103572c1d9ac880ff612fe8df4325109935c2'

def main():
    started=time.monotonic();path=OUT/'holder_supports.npz';metadata=OUT/'holder_supports.json'
    meta=json.loads(metadata.read_text());assert len(meta['parts'])==1330
    protected={str(p.relative_to(ROOT)):H.sha(p) for p in [metadata,OUT/'holder_retained_stumps.npz',
        OUT/'holder_micro_reinforcements.npz',ROOT/'cache/original_v5.npz']}
    for row in meta['parts']:
        for key in ['full_mesh_file','retained_mesh_file','removable_mesh_file']:
            p=OUT/row[key];protected[str(p.relative_to(ROOT))]=H.sha(p)
    before=path.read_bytes();before_hash=hashlib.sha256(before).hexdigest();before_stat=path.stat()
    # Recreate the same constructor order as the last single-part replacement:
    # its old stock, new replacement, then all other full parts in row order.
    row=meta['parts'][1303];old_path=ROOT/row['superseded_precision_fixed_fill']['source_mesh_file']
    old=H.load(old_path)[0];replacement=H.load(OUT/row['full_mesh_file'])[0]
    assert (old-replacement).is_empty()
    pieces=[replacement if r['index']==1303 else H.load(OUT/r['full_mesh_file'])[0] for r in meta['parts']]
    full=H.union(pieces);mesh=full.to_mesh64()
    # Complete all ZIP entries before exposing any bytes at a filesystem path.
    buf=io.BytesIO();np.savez_compressed(buf,vertices=np.asarray(mesh.vert_properties[:,:3]),faces=mesh.tri_verts)
    data=buf.getvalue();digest=hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None and set(z.namelist())=={'vertices.npy','faces.npy'}
    assert all(H.sha(ROOT/p)==d for p,d in protected.items()),'Frozen input changed'
    archive=OUT/'aggregate_recovery_before';archive.mkdir(exist_ok=True)
    (archive/f'holder_supports_{before_hash}.truncated').write_bytes(before)
    temporary=OUT/'holder_supports_recovered.npz.part'
    with temporary.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    assert temporary.read_bytes()==data
    temporary.replace(path)
    directory_fd=os.open(OUT,os.O_RDONLY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    reread=path.read_bytes();assert reread==data and H.sha(path)==digest
    with zipfile.ZipFile(io.BytesIO(reread)) as z:assert z.testzip() is None
    assert all(H.sha(ROOT/p)==d for p,d in protected.items()),'Frozen input changed during restore'
    report=dict(passed=True,scope='Only the regenerable full-holder NPZ was restored; every declared source part and other aggregate is unchanged.',
        part_count=1330,expected_prior_full_sha256=EXPECTED,restored_full_sha256=digest,
        deterministic_expected_bytes_matched=digest==EXPECTED,restored_bytes=len(data),zip_crc_passed=True,
        zip_members=['vertices.npy','faces.npy'],before_sha256=before_hash,before_bytes=len(before),
        before_mtime_ns=before_stat.st_mtime_ns,before_had_end_of_central_directory=b'PK\x05\x06' in before,
        observation='The damaged archive stopped after its first compressed member; no concurrent geometry writer was observed. The underlying cause is undetermined.',
        atomic_write=True,fsync=True,post_rename_bytes_and_hash_verified=True,
        frozen_input_sha256=protected,source_sha256=H.sha(__file__),elapsed_seconds=time.monotonic()-started)
    (ROOT/'reports/holder_full_aggregate_recovery_v6.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frozen_input_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
