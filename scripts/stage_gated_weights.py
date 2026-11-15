#!/usr/bin/env python3
# WHY CUSTOM: huggingface_hub issues file-specific CDN capabilities and SSH transports
# encrypted stdin; neither composes this project's credential-free remote staging policy.
"""Stage the two pinned weight shards without sending an HF account token to a pod."""
from __future__ import annotations

import argparse
import json
import subprocess
from urllib.parse import urlsplit

from huggingface_hub import get_hf_file_metadata, hf_hub_url

REVISION = 'af89d219b53ed9b13b8a4645f9c8028973510324'
SHARDS = {
    'model-00001-of-00002.safetensors': (4992396352, '0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1'),
    'model-00002-of-00002.safetensors': (1706092176, '2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab'),
}
REMOTE = r'''
import concurrent.futures,hashlib,json,os,shutil,sys,urllib.request
from pathlib import Path
payload=json.load(sys.stdin)
root=Path('/workspace/neuroforge/models/tiny-aya-global');root.mkdir(parents=True,exist_ok=True)
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stage(item):
    name=item['filename'];dest=root/name
    try:
        if not dest.exists():
            temporary=root/(name+'.partial')
            with urllib.request.urlopen(item['url'],timeout=120) as response, temporary.open('xb') as target:
                shutil.copyfileobj(response,target,1024*1024);target.flush();os.fsync(target.fileno())
            if temporary.stat().st_size!=item['size'] or digest(temporary)!=item['sha256']:
                raise ValueError('artifact integrity mismatch')
            temporary.rename(dest)
        if dest.stat().st_size!=item['size'] or digest(dest)!=item['sha256']:
            raise ValueError('artifact integrity mismatch')
        return {'filename':name,'sha256':item['sha256'],'verified':True}
    except Exception as error:
        print(json.dumps({'filename':name,'status':'failed','error_type':type(error).__name__}),flush=True)
        raise RuntimeError('staging failed; download capability intentionally not logged') from None
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(stage,payload))
    print(json.dumps({'weights':result,'account_token_transferred':False}),flush=True)
except Exception:
    sys.exit(1)
'''

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--known-hosts', required=True)
    args = parser.parse_args()
    capabilities = []
    for filename, (size, sha) in SHARDS.items():
        metadata = get_hf_file_metadata(hf_hub_url('CohereLabs/tiny-aya-global', filename, revision=REVISION), token=True)
        location = urlsplit(metadata.location)
        if metadata.commit_hash != REVISION or metadata.size != size:
            raise ValueError('pinned artifact metadata mismatch')
        if location.scheme != 'https' or not (location.hostname or '').endswith('.hf.co') or 'Signature=' not in location.query:
            raise ValueError('expected a file-scoped signed HF CDN capability')
        capabilities.append({'filename':filename,'size':size,'sha256':sha,'url':metadata.location})
    command = ['ssh','-i',args.identity,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={args.known_hosts}','-p',str(args.port),f'root@{args.host}','python3','-c',__import__('shlex').quote(REMOTE)]
    result = subprocess.run(command, input=json.dumps(capabilities), text=True, capture_output=True, timeout=900, check=False)
    print(result.stdout, end='')
    if result.returncode:
        print('Remote staging failed; sensitive download capabilities were not logged.')
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
