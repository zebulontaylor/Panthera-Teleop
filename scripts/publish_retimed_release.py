#!/usr/bin/env python3
"""Publish a verified NEW retimed release, verify its hashes, then make public.

Credentials remain in memory. Existing datasets can only be resumed using this
publisher's saved staging record; arbitrary existing releases are never updated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download, get_token

PAYLOAD_ROOTS = {'README.md', 'verification.json', 'data', 'meta', 'processing'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def credential(args):
    token = os.environ.get('HF_TOKEN') or get_token()
    if token:
        return token
    if args.credential_file:
        return read(args.credential_file)['HF_TOKEN']
    if args.credential_host:
        code = 'import json; print(json.load(open("/mnt/pi05/credentials.json"))["HF_TOKEN"])'
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes']
        if args.identity:
            command += ['-i', str(args.identity)]
        command += [args.credential_host, 'python3 -c ' + shlex.quote(code)]
        return subprocess.check_output(command, text=True).strip()
    raise RuntimeError('Sign in with hf auth login or configure an existing credential source')


def inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if relative.parts[0] not in PAYLOAD_ROOTS or '.cache' in relative.parts or '__pycache__' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Published artifacts must be portable files: {relative}')
        if path.is_file() and path.suffix != '.pyc':
            files[relative.as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return files


def lfs_sha(sibling):
    lfs = sibling.lfs
    if not lfs:
        return None
    return lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256


def verify_remote(api, repo, revision, files, token):
    remote = api.dataset_info(repo, revision=revision, files_metadata=True)
    siblings = {s.rfilename: s for s in remote.siblings}
    assert set(files) <= set(siblings), 'Remote files are missing'
    assert not (set(siblings) - set(files) - {'.gitattributes'}), 'Unexpected remote payload'
    lfs_checks = 0
    regular = []
    for name, local in files.items():
        sibling = siblings[name]
        digest = lfs_sha(sibling)
        if digest:
            assert digest == local['sha256'], f'Remote LFS hash differs: {name}'
            assert sibling.size == local['bytes'], f'Remote size differs: {name}'
            lfs_checks += 1
        else:
            regular.append((name, local))
    def verify_regular(item):
        name, local = item
        downloaded = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, token=token))
        assert sha256(downloaded) == local['sha256'], f'Remote file differs: {name}'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_regular, regular))
    return lfs_checks, len(regular)


def publish(args):
    root = args.root.resolve()
    proof = read(root/'verification.json')
    if proof.get('passed') is not True:
        raise ValueError('Successful independent dataset verification is required')
    manifest = read(root/'processing/manifest.json')
    if manifest['repo_id'] != args.repo_id:
        raise ValueError('Dataset manifest repository differs from requested destination')
    if not (root/'README.md').is_file():
        raise ValueError('A dataset card is required before publication')
    token = credential(args)
    api = HfApi(token=token)
    owner = api.whoami()['name']
    if args.repo_id.split('/')[0] != owner:
        raise ValueError('Publishing is restricted to the authenticated owner namespace')
    files = inventory(root)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if args.state.exists():
        state = read(args.state)
        assert state['repo_id'] == args.repo_id and state['inventory_sha256'] == fingerprint
        assert state['root'] == str(root)
        remote = api.dataset_info(args.repo_id)
        if state.get('phase') == 'published':
            revision = state['commit']
        elif remote.sha != state['parent_commit']:
            # Recover an upload whose commit succeeded before the client stopped.
            verify_remote(api, args.repo_id, remote.sha, files, token)
            revision = remote.sha
        else:
            revision = None
    else:
        if api.repo_exists(args.repo_id, repo_type='dataset'):
            raise FileExistsError('Destination exists; choose a new repository')
        api.create_repo(args.repo_id, repo_type='dataset', private=True, exist_ok=False)
        state = dict(repo_id=args.repo_id, root=str(root), phase='staging',
            inventory_sha256=fingerprint, parent_commit=api.dataset_info(args.repo_id).sha,
            created_at=datetime.now(timezone.utc).isoformat())
        write(args.state, state)
        revision = None
    print(f'{args.repo_id}: {len(files)} files, {sum(f["bytes"] for f in files.values())} bytes; staging privately', flush=True)
    if revision is None:
        commit = api.upload_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=root,
            allow_patterns=[f'{name}/**' if name in {'data','meta','processing'} else name for name in PAYLOAD_ROOTS],
            ignore_patterns=['**/.cache/**','**/__pycache__/**','**/*.pyc'],
            parent_commit=state['parent_commit'],
            commit_message=f'Publish verified retimed Panthera dataset ({manifest["totals"]["episodes"]} episodes)')
        revision = commit.oid
        state.update(phase='uploaded', commit=revision)
        write(args.state, state)
    print(f'Verifying remote commit {revision}', flush=True)
    lfs_checks, regular_checks = verify_remote(api, args.repo_id, revision, files, token)
    refs = api.list_repo_refs(args.repo_id, repo_type='dataset')
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    if 'v3.0' in tags:
        assert tags['v3.0'] == revision, 'Existing format tag points to a different commit'
    else:
        api.create_tag(args.repo_id, tag='v3.0', repo_type='dataset', revision=revision)
    assert api.dataset_info(args.repo_id, revision='v3.0').sha == revision
    api.update_repo_settings(args.repo_id, repo_type='dataset', private=False)
    public = HfApi(token=False).dataset_info(args.repo_id, revision=revision)
    assert public.private is False and public.sha == revision
    state.update(phase='published', public=True, commit=revision,
        url=f'https://huggingface.co/datasets/{args.repo_id}',
        remote_files_verified=len(files), remote_lfs_sha256_files_verified=lfs_checks,
        remote_regular_sha256_files_verified=regular_checks,
        episodes=manifest['totals']['episodes'], frames=manifest['totals']['observations'])
    write(args.state, state)
    write(root/'publication.json', {k:v for k,v in state.items() if k not in {'parent_commit','inventory_sha256'}})
    print(json.dumps({k:v for k,v in state.items() if k not in {'parent_commit','inventory_sha256'}}, indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--credential-host')
    p.add_argument('--identity',type=Path)
    p.add_argument('--credential-file',type=Path)
    publish(p.parse_args())
