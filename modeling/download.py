"""Download public IDSSE files only, atomically, with provenance and hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MATCHES = {m: ('000001' if m in ('J03WMX', 'J03WN1') else '000002')
           for m in ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(match, folder):
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, prefix in [('meta', 'DFL_02_01_matchinformation'),
                         ('tracking', 'DFL_04_03_positions_raw_observed')]:
        filename = f'{prefix}_DFL-COM-{MATCHES[match]}_DFL-MAT-{match}.xml'
        url = f'https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{filename}'
        dest = folder / f'{match}-{kind}.xml'
        if not dest.exists():
            tmp = dest.with_suffix('.part')
            print(f'Downloading {match} {kind}', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, 'wb') as out:
                expected = response.headers.get('Content-Length')
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    out.write(block)
                    total += len(block)
                if expected is not None and total != int(expected):
                    raise IOError('Truncated download; incomplete .part file retained')
            tmp.replace(dest)
        with open(dest, 'rb') as f:
            f.seek(max(0, dest.stat().st_size-256))
            if b'</PutDataRequest>' not in f.read():
                raise IOError(f'{dest} is not a complete IDSSE XML file; do not train on it')
        records.append(dict(path=dest.name, url=url, bytes=dest.stat().st_size,
                            sha256=sha256(dest)))
    (folder / f'{match}-provenance.json').write_text(json.dumps({
        'dataset': 'IDSSE, Bassek et al. 2025', 'license': 'CC-BY-4.0',
        'paper': 'https://doi.org/10.1038/s41597-025-04505-y', 'files': records
    }, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches', nargs='+', choices=MATCHES, required=True)
    p.add_argument('--out', type=Path, default=Path('data-external'))
    a = p.parse_args()
    for match in a.matches:
        download(match, a.out)
