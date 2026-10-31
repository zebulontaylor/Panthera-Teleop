#!/usr/bin/env python3
"""Add explicit viewer file selection to this task's already-published release.

All previously verified remote blobs must remain unchanged except README.md.
"""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from publish_retimed_release import credential, inventory, read, write
import json


def viewer_card(text):
    if '\nconfigs:' in text:
        return text
    if not text.startswith('---\n') or '\n---\n' not in text[4:]:
        raise ValueError('Expected YAML dataset card header')
    end=text.index('\n---\n',4)
    return text[:end]+'''\nconfigs:
- config_name: default
  data_files:
  - split: train
    path: data/chunk-*/*.parquet'''+text[end:]


def main(args):
    state=read(args.state)
    assert state['phase']=='published' and state['public'] is True
    root=args.root.resolve();repo=state['repo_id'];old_revision=state['commit']
    assert read(root/'processing/manifest.json')['repo_id']==repo
    token=credential(args);api=HfApi(token=token)
    old=api.dataset_info(repo,revision=old_revision,files_metadata=True)
    assert api.dataset_info(repo).sha==old_revision
    path=root/'README.md';card=viewer_card(path.read_text())
    remote_path=Path(hf_hub_download(repo,'README.md',repo_type='dataset',revision=old_revision,token=token))
    if remote_path.read_text()==card:
        print('Viewer config already present',flush=True);return
    commit=api.upload_file(path_or_fileobj=card.encode(),path_in_repo='README.md',repo_id=repo,
        repo_type='dataset',parent_commit=old_revision,
        commit_message='Select training Parquet files explicitly for the dataset viewer')
    new=api.dataset_info(repo,revision=commit.oid,files_metadata=True)
    before={s.rfilename:s.blob_id for s in old.siblings}
    after={s.rfilename:s.blob_id for s in new.siblings}
    assert before.keys()==after.keys()
    assert {k for k in before if before[k]!=after[k]}=={'README.md'}
    downloaded=Path(hf_hub_download(repo,'README.md',repo_type='dataset',revision=commit.oid,token=token))
    assert hashlib.sha256(downloaded.read_bytes()).digest()==hashlib.sha256(card.encode()).digest()
    path.write_text(card)
    api.delete_tag(repo,repo_type='dataset',tag='v3.0')
    api.create_tag(repo,repo_type='dataset',tag='v3.0',revision=commit.oid)
    assert HfApi(token=False).dataset_info(repo,revision='v3.0').sha==commit.oid
    files=inventory(root)
    state.update(payload_commit=old_revision,commit=commit.oid,viewer_training_files_explicit=True,
        unchanged_remote_blobs_except_readme_verified=True,
        inventory_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    write(args.state,state)
    write(root/'publication.json',{k:v for k,v in state.items() if k not in {'parent_commit','inventory_sha256'}})
    print(f'{repo}: viewer config verified at {commit.oid}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--credential-host')
    p.add_argument('--identity',type=Path)
    p.add_argument('--credential-file',type=Path)
    main(p.parse_args())
