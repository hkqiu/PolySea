"""Pack the exact 20 auxiliary training CSVs into a deterministic gzip tar.

The separately released merged_train.csv is excluded; its exact hash is still
recorded in source_file_manifest.json. Nothing in the source directory is edited.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must not exist')
    entries = json.loads(args.manifest.read_text(encoding='utf-8'))
    selected = [entry for entry in entries if entry['file'] != 'merged_train.csv']
    if len(entries) != 21 or len(selected) != 20:
        raise ValueError('Expected the archived manifest of 21 training CSVs')
    contents = []
    for entry in sorted(selected, key=lambda value: value['file']):
        name = entry['file']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Manifest filenames must be simple basenames')
        raw = (args.train_dir / name).read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        contents.append((name, raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                for name, raw in contents:
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(raw), 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    archive.addfile(info, io.BytesIO(raw))
    print(json.dumps({'members': len(contents), 'bytes': args.output.stat().st_size,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
