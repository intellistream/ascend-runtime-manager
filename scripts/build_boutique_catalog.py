#!/usr/bin/env python3
"""Build pinned official catalog with a documented loopback-only listener patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

SHA = 'b9a978db9e01f4ad3dca9494a22cb9edc17548fe'
p = argparse.ArgumentParser()
p.add_argument('--app', required=True, type=Path)
p.add_argument('--go', required=True, type=Path)
p.add_argument('--state', required=True, type=Path)
a = p.parse_args()
a.app, a.go, a.state = a.app.resolve(), a.go.resolve(), a.state.resolve()
assert subprocess.check_output(['git', '-C', str(a.app), 'rev-parse', 'HEAD'], text=True).strip() == SHA
relative = 'src/productcatalogservice/server.go'
original = subprocess.check_output(['git', '-C', str(a.app), 'show', f'HEAD:{relative}'], text=True)
old, new = 'fmt.Sprintf(":%s", port)', 'fmt.Sprintf("127.0.0.1:%s", port)'
assert original.count(old) == 1
patched = original.replace(old, new)
source = a.app / relative
assert source.read_text() in (original, patched), 'refusing to replace unrelated source changes'
source.write_text(patched)
a.state.mkdir(parents=True, exist_ok=True)
(a.state / 'bin').mkdir(exist_ok=True)
env = {**os.environ, 'GOMAXPROCS': '2', 'GOTOOLCHAIN': 'local',
       'GOPATH': str(a.state / 'go-path'), 'GOCACHE': str(a.state / 'go-cache')}
binary = a.state / 'bin/productcatalog'
subprocess.run([str(a.go), '-C', str(source.parent), 'build', '-p', '2', '-o', str(binary), '.'],
               env=env, check=True)
receipt = {'source_sha': SHA, 'patch': {'before': old, 'after': new},
           'go_version': subprocess.check_output([str(a.go), 'version'], text=True).strip(),
           'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
           'go_sum_sha256': hashlib.sha256((source.parent/'go.sum').read_bytes()).hexdigest(),
           'goproxy': env.get('GOPROXY', 'default'), 'gosumdb': env.get('GOSUMDB', 'default')}
(a.state/'catalog-build-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
