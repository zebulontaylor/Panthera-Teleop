#!/usr/bin/env python3
"""Publish a verified processed dataset, then verify the remote release."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main(root):
    verification = json.loads((root / "verification.json").read_text())
    assert verification["passed"]
    assert json.loads((root / "processing/full_rate_verification.json").read_text())["passed"]
    manifest = json.loads((root / "processing/manifest.json").read_text())
    repo_id = manifest["repo_id"]
    api = HfApi()
    assert api.whoami()["name"] == repo_id.split("/")[0]
    # Avoid replacing any existing user dataset. Create privately until upload
    # and remote checks are complete, then publish the authorized release.
    url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
    print(f"Created staging dataset {url}", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=root,
        allow_patterns=["README.md", "crop_preview.jpg", "verification.json",
                        "data/**", "meta/**", "processing/**", "trimmed_full_rate/**"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.cache/**"],
        commit_message="Publish 30 contact-trimmed real Panthera assembly demos, square crops, 20 Hz LeRobot v3")
    revision = commit.oid
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    expected = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                and p.parts[len(root.parts)] in
                ("README.md", "crop_preview.jpg", "verification.json", "data", "meta", "processing", "trimmed_full_rate")
                and "__pycache__" not in p.parts and p.suffix != ".pyc" and ".cache" not in p.parts]
    assert not (set(expected) - set(files)), sorted(set(expected) - set(files))
    for name in ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
                 "processing/manifest.json", "verification.json"]:
        remote = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name, revision=revision))
        assert hashlib.sha256(remote.read_bytes()).digest() == hashlib.sha256((root / name).read_bytes()).digest()
    api.create_tag(repo_id=repo_id, repo_type="dataset", tag="v3.0", revision=revision)
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=False)
    public = HfApi(token=False).dataset_info(repo_id, revision=revision)
    assert not public.private
    result = dict(repo_id=repo_id, url=str(url), commit=revision, public=True,
                  remote_files_verified=len(expected), remote_metadata_hashes_verified=True)
    (root.parent / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root.resolve())
