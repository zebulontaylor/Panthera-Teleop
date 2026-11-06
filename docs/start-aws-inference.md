# Start AWS inference

From the repository root:

```bash
python3 scripts/start-aws-inference.py --dry-run
python3 scripts/start-aws-inference.py
# Restrict discovery to the intended training run:
python3 scripts/start-aws-inference.py 54.161.106.103 --run-dir /mnt/pi05/runs/panthera-pi05-dtw80-mse20-666-20261031
```

Defaults: user `ubuntu`, identity `~/.ssh/sshkey.pem`. Discovery selects the newest complete numeric checkpoint by weights modification time across `/mnt/pi05/runs`, excluding smoke runs and moving `last` links. Dry run reads the dashboard and remote metadata, validates the contract, and makes no deployment changes (SSH may record a newly accepted host key).

The script requires an idle dashboard on port 8093 before changing services. It validates local/remote contracts, absolute actions, task and camera crops. Missing `tasks` may be populated from `training_tasks` only on the inference copy; training metadata stays unchanged. It reuses a matching active snapshot, or copies and verifies a new snapshot outside trainer pruning. New service configuration explicitly uses `--allow-training --cuda-memory-fraction 0.16`. Existing override and tunnel files receive timestamped backups before replacement.

The local user tunnel forwards `127.0.0.1:8766` to remote `127.0.0.1:8765`. Readiness has a 300-second deadline and checks checkpoint and contract identity. The existing local inference probe verifies a synthetic prediction without robot motion. A connected holding dashboard refreshes policy with `/api/recover`, retaining backend hold. A disconnected dashboard stays disconnected. No dashboard or motor service is restarted. Use `--dashboard-url` to select another loopback dashboard, or `--skip-dashboard-refresh` to leave its policy session unchanged after the safety check.

Failures leave diagnostics and backups in place; the script does not automatically roll back. Inspect `systemctl --user cat panthera-inference` on AWS and `systemctl --user cat panthera-inference-tunnel` locally before restoring a timestamped backup and reloading systemd. Snapshots consume approximately 9 GB each.
