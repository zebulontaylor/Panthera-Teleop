#!/usr/bin/env bash
# Stage only a compact state-critic dataset and one trainer script, then start
# an explicitly requested detached AWS run under a new, isolated run directory.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/state_critic_aws.sh [--launch] DATASET_DIR TRAINER_SCRIPT OUTPUT_NAME

Without --launch this prints the selected inputs and performs no network or
filesystem writes. With --launch it first runs a read-only AWS preflight, then
stages samples.jsonl plus only referenced images, uploads one trainer script,
and starts a detached 40-epoch, batch-8 run.

AWS access uses ~/.ssh/sshkey.pem and admin@54.166.156.98. The remote run root
is /mnt/pi05/state-critic-20261028. Existing outputs are never overwritten.
EOF
}

launch=0
if [[ ${1:-} == --launch ]]; then
  launch=1
  shift
fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi

dataset_dir=$(realpath "$1")
trainer_script=$(realpath "$2")
output_name=$3
if [[ ! -d $dataset_dir || ! -f $dataset_dir/samples.jsonl ]]; then
  echo "Dataset must be a directory containing samples.jsonl: $dataset_dir" >&2; exit 2
fi
if [[ ! -f $trainer_script || ! -r $trainer_script ]]; then
  echo "Trainer script is not a readable file: $trainer_script" >&2; exit 2
fi
if [[ ! $output_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "OUTPUT_NAME must be 1-64 safe filename characters" >&2; exit 2
fi

host=admin@54.166.156.98
key="$HOME/.ssh/sshkey.pem"
remote_root=/mnt/pi05/state-critic-20261028
remote_input="$remote_root/input/$output_name"
remote_run="$remote_root/runs/$output_name"
remote_script="$remote_root/trainer-$output_name.py"
if [[ ! -r $key ]]; then echo "SSH identity is unavailable: $key" >&2; exit 2; fi

if (( ! launch )); then
  printf 'Validated local dataset: %s\nTrainer: %s\nOutput name: %s\nRemote host: %s\nRemote input: %s\nRemote run: %s\n' \
    "$dataset_dir" "$trainer_script" "$output_name" "$host" "$remote_input" "$remote_run"
  echo 'No network access or writes performed. Add --launch only after reviewing this plan.'
  exit 0
fi

ssh_args=(-i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)

# Do not upload a dataset that cannot support the trainer's supervised objective.
python3 - "$dataset_dir/samples.jsonl" <<'LOCAL_LABEL_PREFLIGHT'
import json, sys
classes = set()
with open(sys.argv[1]) as stream:
    for line in stream:
        row = json.loads(line)
        if row.get('split') == 'train' and row.get('label_source', 'reviewed') == 'reviewed':
            if row.get('outcome') in (0, 1):
                classes.add(row['outcome'])
if classes != {0, 1}:
    raise SystemExit('Training not launched: reviewed training labels need both outcomes; unknown or curation-proxy labels cannot substitute for verified coverage.')
LOCAL_LABEL_PREFLIGHT

# The remote probe is intentionally read-only. Existing GPU processes are
# reported but permitted when at least 4 GiB are free, as authorized.
set +e
preflight=$(ssh "${ssh_args[@]}" "$host" bash -s -- "$remote_root" "$remote_input" "$remote_run" "$remote_script" <<'REMOTE_PREFLIGHT'
set -euo pipefail
root=$1; input=$2; run=$3; script=$4
if ! command -v nvidia-smi >/dev/null; then echo 'UNAVAILABLE: nvidia-smi missing' >&2; exit 21; fi
gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
disk=$(df -Pm /mnt/pi05 | awk 'NR==2 {print $4}')
printf 'GPU name,total_mib,free_mib,util_percent: %s\nDisk free MiB: %s\n' "$gpu" "$disk"
printf 'Active GPU processes:\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
printf 'Candidate related processes:\n'
pgrep -af 'state_critic|state-critic|train_panthera|lerobot' || true
python=/mnt/pi05/lerobot/.venv/bin/python
if [[ ! -x $python ]]; then echo 'UNAVAILABLE: pinned Python runtime missing' >&2; exit 22; fi
"$python" - <<'PY'
import sys
try:
    import numpy, PIL, sklearn, torch, torchvision
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=None)
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access the CUDA GPU")
    print(f'Python/dependencies OK: python={sys.version.split()[0]} torch={torch.__version__} torchvision={torchvision.__version__} CUDA_build={torch.version.cuda} cuda_available={torch.cuda.is_available()} numpy={numpy.__version__} sklearn={sklearn.__version__} PIL={PIL.__version__} resnet18={type(model).__name__} weights={ResNet18_Weights.DEFAULT.name}')
except Exception as exc:
    print(f'UNAVAILABLE: runtime dependency/torchvision probe failed: {type(exc).__name__}: {exc}', file=sys.stderr)
    raise SystemExit(23)
PY
if [[ ! $free_mib =~ ^[0-9]+$ ]] || (( free_mib < 4096 )); then echo 'UNAVAILABLE: less than 4 GiB GPU memory free' >&2; exit 24; fi
if [[ ! $disk =~ ^[0-9]+$ ]] || (( disk < 5120 )); then echo 'UNAVAILABLE: less than 5 GiB free on /mnt/pi05' >&2; exit 25; fi
if [[ -e $input || -e $run || -e $script ]]; then echo 'UNAVAILABLE: remote input, run, or trainer path already exists; refusing overwrite' >&2; exit 26; fi
REMOTE_PREFLIGHT
)
preflight_rc=$?
set -e
if (( preflight_rc != 0 )); then
  if (( preflight_rc == 255 )); then echo 'AWS status: unreachable; no remote mutation performed.' >&2
  else echo "AWS preflight failed (status $preflight_rc); no remote mutation performed." >&2; fi
  [[ -z $preflight ]] || printf '%s\n' "$preflight" >&2
  exit "$preflight_rc"
fi
printf '%s\n' "$preflight"

stage=$(mktemp -d "${TMPDIR:-/tmp}/state-critic-aws.XXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT
mkdir -p "$stage/dataset/images"

# Copy exactly the manifest and image files it references. This excludes audit
# records, proxy labels, raw episodes, and unrelated files in the source tree.
python3 - "$dataset_dir" "$stage/dataset" <<'PY'
import json, pathlib, shutil, sys
source, stage = map(pathlib.Path, sys.argv[1:])
manifest = source / "samples.jsonl"
seen = set()
camera_order = ("overhead", "wrist_port2", "wrist_port3")
with manifest.open(encoding="utf-8") as src, (stage / "samples.jsonl").open("w", encoding="utf-8") as dst:
    for number, line in enumerate(src, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        refs = []
        for key in ("images", "history_images"):
            images = row.get(key)
            if images is None and key == "history_images":
                continue
            if isinstance(images, dict):
                if set(images) == set(camera_order):
                    row[key] = [images[camera] for camera in camera_order]
                else:
                    raise SystemExit(f"manifest row {number}: expected camera keys {camera_order}")
                refs.extend(row[key])
            elif isinstance(images, list):
                refs.extend(images)
            else:
                raise SystemExit(f"manifest row {number}: invalid {key}")
        if not refs:
            raise SystemExit(f"manifest row {number}: no referenced images")
        for rel in refs:
            path = pathlib.PurePosixPath(str(rel))
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"manifest row {number}: unsafe image reference {rel!r}")
            if rel in seen:
                continue
            original = source.joinpath(*path.parts)
            if not original.is_file():
                raise SystemExit(f"manifest row {number}: missing image {original}")
            target = stage.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            seen.add(rel)
        if row.get("split") == "validation":
            row["split"] = "val"
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Staged {len(seen)} referenced images and samples.jsonl")
PY

incoming="$remote_root/input/.${output_name}.incoming-$$"
if ! ssh "${ssh_args[@]}" "$host" bash -s -- "$remote_root" "$incoming" "$remote_input" "$remote_run" "$remote_script" <<'REMOTE_PREPARE'
set -euo pipefail
root=$1; incoming=$2; input=$3; run=$4; script=$5
[[ ! -e $incoming && ! -e $input && ! -e $run && ! -e $script ]] || { echo 'Remote target appeared after preflight; refusing overwrite' >&2; exit 30; }
mkdir -p "$root/input" "$root/runs"
mkdir "$incoming"
REMOTE_PREPARE
then
  echo 'Remote staging preparation failed; no training launched.' >&2; exit 1
fi

if ! rsync -a --delete "$stage/dataset/" -e "ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30" "$host:$incoming/"; then
  echo 'Upload failed; staged data remains under its isolated incoming path for inspection.' >&2
  exit 1
fi

if ! scp "${ssh_args[@]}" "$trainer_script" "$host:$remote_script"; then
  echo 'Trainer upload failed; training was not launched.' >&2; exit 1
fi

# Recheck GPU and disk immediately before starting. Active processes remain
# allowed when the minimum free resources still exist.
ssh "${ssh_args[@]}" "$host" bash -s -- "$remote_root" "$incoming" "$remote_input" "$remote_run" "$remote_script" "$output_name" <<'REMOTE_LAUNCH'
set -euo pipefail
root=$1; incoming=$2; input=$3; run=$4; script=$5; name=$6
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
disk=$(df -Pm /mnt/pi05 | awk 'NR==2 {print $4}')
[[ $free_mib =~ ^[0-9]+$ ]] && (( free_mib >= 4096 )) || { echo 'GPU memory dropped below 4 GiB; refusing launch' >&2; exit 31; }
[[ $disk =~ ^[0-9]+$ ]] && (( disk >= 5120 )) || { echo 'Disk free space dropped below 5 GiB; refusing launch' >&2; exit 32; }
[[ ! -e $input && ! -e $run ]] || { echo 'Remote destination appeared; refusing overwrite' >&2; exit 33; }
python=/mnt/pi05/lerobot/.venv/bin/python
help=$("$python" "$script" --help 2>&1)
args=(--manifest "$input/samples.jsonl" --dataset-root "$input" --output "$run/output" --epochs 40 --batch-size 8 --feature-batch-rows 2)
memory_fraction=0
if grep -q -- '--memory-fraction' <<<"$help"; then memory_fraction=1; fi
mkdir -p "$run"
mv "$incoming" "$input"
cat > "$run/launch.sh" <<EOF
#!/usr/bin/env bash
set +e
printf 'running\\n' > '$run/status'
args=(--manifest '$input/samples.jsonl' --dataset-root '$input' --output '$run/output' --epochs 40 --batch-size 8 --feature-batch-rows 2)
if [[ '$memory_fraction' == 1 ]]; then args+=(--memory-fraction 0.1); fi
'$python' '$script' "\${args[@]}"
rc=\$?
if (( rc == 0 )); then printf 'complete\\n' > '$run/status'; else printf 'failed:%s\\n' "\$rc" > '$run/status'; fi
exit "\$rc"
EOF
chmod 700 "$run/launch.sh"
nohup bash "$run/launch.sh" > "$run/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/pid"
printf 'launched pid=%s\nrun=%s\ntrainer=%s\npython=%s\n' "$pid" "$run" "$script" "$python"
REMOTE_LAUNCH

echo 'Detached run requested. Inspect remotely with: cat /mnt/pi05/state-critic-20261028/runs/'"$output_name"'/status; tail -f /mnt/pi05/state-critic-20261028/runs/'"$output_name"'/train.log'
