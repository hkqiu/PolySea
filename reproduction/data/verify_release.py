"""Verify the published release allowlist, checksums, CSV counts and tar members.

Uses only the Python standard library and never extracts or modifies files.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def check(raw, entry):
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Byte count or SHA256 mismatch')
    if 'data_rows' in entry:
        reader = csv.reader(io.StringIO(raw.decode(entry['encoding']), newline=''))
        header = next(reader)
        if header != entry['columns'] or sum(1 for _ in reader) != entry['data_rows']:
            raise ValueError('CSV header or data-record count mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = json.loads((root / 'dataset/polyopus/release_manifest.json').read_text(encoding='utf-8'))
    member_count = 0
    for entry in manifest['files']:
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path = root / relative
        raw = path.read_bytes()
        try:
            check(raw, entry)
            if 'gzip_content' in entry:
                check(gzip.decompress(raw), entry['gzip_content'])
            if 'tar_members' in entry:
                with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
                    expected = {member['file']: member for member in entry['tar_members']}
                    members = archive.getmembers()
                    if len(members) != len(expected) or {member.name for member in members} != set(expected):
                        raise ValueError('Tar member allowlist mismatch')
                    for member in members:
                        if not member.isfile():
                            raise ValueError('Non-regular archive member')
                        check(archive.extractfile(member).read(), expected[member.name])
                        member_count += 1
        except Exception as exc:
            raise ValueError(f'Verification failed for {entry["path"]}: {exc}') from exc
    print(json.dumps({'verified_release_files': len(manifest['files']),
                      'verified_training_archive_members': member_count,
                      'all_checks_passed': True}))


if __name__ == '__main__':
    main()
