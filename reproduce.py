"""Verify the frozen source/model/input contract, then reproduce the exact CSV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED='bc4320370b362aa195c0b99bd39fe1137abc42290e09a3c4293f1a7f1d61a1b1'


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-repo',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('outputs/submission'))
    a=p.parse_args()
    if Path.cwd().resolve()!=root: raise SystemExit('Run from the extracted source repository directory')
    if sys.version_info[:2]!=(3,12) or os.name!='nt':
        raise SystemExit('This frozen release is verified for Windows/Python3.12; other platforms are unverified')
    # File checks precede loading our trusted local serialized models.
    subprocess.run([sys.executable,'verify_release.py'],check=True)
    from modeling.predict_frozen_target import source_paths
    contract=json.loads((root/'input-contract.json').read_text())
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths(a.assets_repo.resolve())}
    if actual!=contract['input_sha256_by_filename']:
        raise SystemExit('Original organizer code/ATD input hashes differ from the frozen input contract')
    subprocess.run([sys.executable,'-u','-m','modeling.predict_frozen_target','--candidate','candidate/manifest.json',
        '--repo',str(a.assets_repo.resolve()),'--out',str(a.out)],check=True)
    output=a.out/'possession.csv'
    if hashlib.sha256(output.read_bytes()).hexdigest()!=EXPECTED: raise SystemExit('CSV differs from the frozen submission hash')
    print('Reproduced exact199319-rowCSV:',output)


if __name__=='__main__': main()
