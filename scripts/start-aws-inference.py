#!/usr/bin/env python3
"""Select, validate and serve a complete AWS checkpoint without touching robot control."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('model.safetensors', 'config.json', 'deployment.json', 'policy_preprocessor.json', 'policy_postprocessor.json')


def select_checkpoint(runs, run_dir=None):
    """Choose newest completed numeric checkpoint by weights mtime, excluding smoke runs."""
    candidates = []
    roots = [Path(run_dir)] if run_dir else Path(runs).iterdir()
    for run in roots:
        if 'smoke' in run.name.lower():
            continue
        checkpoints = run / 'checkpoints'
        if not checkpoints.is_dir():
            continue
        for step in checkpoints.iterdir():
            model = step / 'pretrained_model'
            if not step.name.isdigit() or step.is_symlink():
                continue
            if all((model / name).is_file() and (model / name).stat().st_size for name in REQUIRED):
                try:
                    json.loads((model / 'config.json').read_text())
                    json.loads((model / 'deployment.json').read_text())
                except (ValueError, OSError):
                    continue
                candidates.append((model.joinpath('model.safetensors').stat().st_mtime_ns,
                                   int(step.name), str(model)))
    if not candidates:
        raise ValueError('No complete numeric checkpoint found')
    return Path(max(candidates)[2])


def normalize_metadata(meta):
    meta = dict(meta)
    if 'task' not in meta and not meta.get('tasks') and meta.get('training_tasks'):
        meta['tasks'] = meta['training_tasks']
    return meta


def validate_contract(meta, config, contract):
    for key in ('fps', 'dataset_repo', 'dataset_revision', 'dataset_manifest_sha256', 'dataset_mix'):
        if key in contract and meta.get(key) != contract[key]:
            raise ValueError(f'Checkpoint deployment mismatch: {key}')
    tasks = [meta['task']] if 'task' in meta else meta.get('tasks', [])
    if (not isinstance(tasks, list) or not tasks or
            any(not isinstance(task, str) or not task.strip() for task in tasks) or
            contract['task'] not in tasks):
        raise ValueError('Checkpoint deployment mismatch: task')
    if meta.get('action_representation') != 'absolute' or config.get('use_relative_actions') is not False:
        raise ValueError('Only explicitly absolute joint actions are supported')
    crops = {f'observation.images.{name}': crop for name, crop in contract['image_crops_xyxy'].items()}
    if meta.get('image_crops') != crops:
        raise ValueError('Checkpoint deployment mismatch: image_crops')


def validate_target(host, user):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) or '..' in host:
        raise ValueError('Host must be a hostname or IPv4 address without SSH options')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', user):
        raise ValueError('Invalid SSH user')


def remote_command(payload):
    return 'python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(json.dumps(payload))


# Discovery uses the exact same helpers locally and remotely; no shell interpolation.
REMOTE = '''import json,sys,os,shutil,subprocess,time,hashlib
from pathlib import Path
p=json.loads(sys.argv[1])
'''
import inspect
REMOTE += 'REQUIRED=' + repr(REQUIRED) + '\n' + inspect.getsource(select_checkpoint) + '\n' + inspect.getsource(normalize_metadata)
REMOTE += '''
if p['action']=='discover':
    model=select_checkpoint('/mnt/pi05/runs',p.get('run_dir'))
    print(json.dumps(dict(checkpoint=str(model),deployment=json.loads((model/'deployment.json').read_text()),config=json.loads((model/'config.json').read_text()),contract=json.loads(Path('/mnt/pi05/inference/contract.json').read_text()))))
else:
    source=Path(p['checkpoint'])
    # Reuse the running immutable snapshot when it has the exact selected weights.
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5) as response: current=json.load(response)
        existing=Path(current.get('checkpoint','/nonexistent'))
        def same_weights(a,b):
            if not a.is_file() or not b.is_file() or a.stat().st_size!=b.stat().st_size: return False
            if os.path.samefile(a,b): return True
            def sha(path):
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
                return h.digest()
            return sha(a)==sha(b)
        if (current.get('status') in ('ready','loading') and existing.is_relative_to('/mnt/pi05/inference/checkpoints')
            and current.get('contract')==json.loads(p['remote_contract_text'])
            and same_weights(source/'model.safetensors',existing/'model.safetensors')
            and json.loads((existing/'config.json').read_text())==json.loads((source/'config.json').read_text())
            and normalize_metadata(json.loads((existing/'deployment.json').read_text()))==normalize_metadata(json.loads((source/'deployment.json').read_text()))):
            print(json.dumps({'checkpoint':str(existing),'reused':True}))
            sys.exit(0)
    except (OSError,ValueError): pass
    if not all(c.isalnum() or c in '-_.' for c in source.parents[2].name): raise ValueError('Unsupported run directory name')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    snapshot=Path('/mnt/pi05/inference/checkpoints')/(source.parents[2].name+'-'+source.parent.name+'-'+stamp)
    if snapshot.exists(): raise RuntimeError('Snapshot already exists')
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    # Copy rather than linking: trainer pruning/replacement cannot mutate this snapshot.
    shutil.copytree(source,snapshot,symlinks=False)
    for name in REQUIRED:
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            return h.digest()
        if digest(source/name)!=digest(snapshot/name): raise RuntimeError('Checkpoint changed during copy')
    metadata=snapshot/'deployment.json'
    metadata.write_text(json.dumps(normalize_metadata(json.loads(metadata.read_text())),indent=2)+'\\n')
    for path in snapshot.rglob('*'):
        if path.is_file(): path.chmod(path.stat().st_mode & ~0o222)
    drop=Path.home()/'.config/systemd/user/panthera-inference.service.d'
    drop.mkdir(parents=True,exist_ok=True)
    target=drop/'zz-start-aws.conf'
    if target.exists(): shutil.copy2(target,str(target)+'.bak-'+stamp)
    if Path('/mnt/pi05/inference/contract.json').read_text()!=p['remote_contract_text']:
        raise RuntimeError('Remote contract changed after validation')
    # Explicit runtime configuration; never execute a discovered service command.
    target.write_text('[Service]\\nExecStart=\\nExecStart=/mnt/pi05/lerobot/.venv/bin/python /mnt/pi05/inference/server.py --checkpoint "'+str(snapshot)+'" --allow-training --cuda-memory-fraction 0.16\\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','panthera-inference.service'],check=True,stdout=sys.stderr)
    subprocess.run(['systemctl','--user','restart','panthera-inference.service'],check=True)
    print(json.dumps({'checkpoint':str(snapshot)}))
'''


def health(url, checkpoint, contract, timeout):
    deadline = time.monotonic() + timeout
    last = 'No response'
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + '/health', timeout=5) as response:
                value = json.load(response)
            if value.get('status') in ('failed', 'error'):
                raise RuntimeError('Inference load failed: '+str(value))
            if value.get('status') == 'ready':
                if value.get('checkpoint') != checkpoint or value.get('contract') != contract or value.get('mock'):
                    raise ValueError('Ready server checkpoint/contract identity mismatch')
                return value
            last = str(value)
        except (OSError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError('Inference did not become ready: ' + last)


def dashboard_status(url):
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('Dashboard URL must use loopback HTTP')
    try:
        with urllib.request.urlopen(url + '/api/status', timeout=5) as response:
            return json.load(response)
    except OSError as exc:
        raise RuntimeError('Cannot check dashboard safety; start dashboard or check --dashboard-url: '+str(exc))


def ensure_dashboard_idle(state):
    if state.get('playing') or state.get('rollout_active') or state.get('collection_pending'):
        raise ValueError('Pause dashboard and finish collection before changing inference')
    if state.get('connected') and state.get('mode') != 'holding':
        raise ValueError('Connected dashboard must be holding before changing inference')


def refresh_dashboard(url, checkpoint):
    state = dashboard_status(url)
    ensure_dashboard_idle(state)
    if not state.get('connected'):
        print('Dashboard disconnected: endpoint ready; policy loads on next manual connect.')
        return
    if state.get('policy_ready') and state.get('checkpoint') == checkpoint:
        print('Dashboard already uses selected checkpoint.')
        return
    request = urllib.request.Request(url+'/api/recover', data=b'{}',
        headers={'Content-Type':'application/json','X-Session-Token':state['token']},method='POST')
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    deadline = time.monotonic()+60
    while time.monotonic()<deadline:
        state=dashboard_status(url)
        if (state.get('policy_ready') and state.get('checkpoint')==checkpoint
                and state.get('connected') and state.get('mode')=='holding'
                and state.get('hold_status')=='commanded' and not state.get('error')):
            print('Dashboard policy refreshed; backend retains hold.')
            return
        time.sleep(1)
    raise RuntimeError('Endpoint ready but dashboard policy refresh did not complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', default='54.161.106.103')
    parser.add_argument('--user', default='ubuntu')
    parser.add_argument('--identity', type=Path, default=Path.home()/'.ssh/sshkey.pem')
    parser.add_argument('--run-dir', help='Restrict selection to this run; default searches all non-smoke runs')
    parser.add_argument('--dashboard-url', default='http://127.0.0.1:8093')
    parser.add_argument('--skip-dashboard-refresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Read-only discovery and validation')
    args = parser.parse_args()
    validate_target(args.host, args.user)
    identity = str(args.identity.expanduser().resolve())
    if not Path(identity).is_file() or any(c in identity for c in '\\"%\n\r'):
        raise ValueError('Missing identity or unsupported systemd identity path')
    dashboard = dashboard_status(args.dashboard_url)
    ensure_dashboard_idle(dashboard)
    ssh = ['/usr/bin/ssh', '-T', '-i', str(args.identity.expanduser()), '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=3',
           '-o', 'StrictHostKeyChecking=accept-new', f'{args.user}@{args.host}']
    def remote(payload):
        return json.loads(subprocess.check_output(ssh + [remote_command(payload)], text=True))
    selected = remote(dict(action='discover', run_dir=args.run_dir))
    contract = json.loads((ROOT/'infra/inference/contract.json').read_text())
    if selected['contract'] != contract:
        raise ValueError('Remote inference contract differs from local contract')
    validate_contract(normalize_metadata(selected['deployment']), selected['config'], contract)
    print('Selected:', selected['checkpoint'], flush=True)
    print('Validated absolute actions, crops, task and dataset contract.', flush=True)
    if args.dry_run:
        print('Dry run: no files or services changed.')
        return
    # Exact text avoids silently changing or replacing a validated remote contract.
    remote_text = subprocess.check_output(ssh + ['cat /mnt/pi05/inference/contract.json'], text=True)
    if json.loads(remote_text) != contract:
        raise ValueError('Remote contract changed after discovery')
    result = remote(dict(action='apply', checkpoint=selected['checkpoint'], remote_contract_text=remote_text))
    unit = Path.home()/'.config/systemd/user/panthera-inference-tunnel.service'
    unit.parent.mkdir(parents=True, exist_ok=True)
    if unit.exists():
        import shutil
        shutil.copy2(unit, str(unit)+'.bak-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    # systemd's quoting differs from shell quoting; reject unsupported identity characters.
    identity = str(args.identity.expanduser().resolve())
    if any(c in identity for c in '\\"%\n\r'):
        raise ValueError('Identity path contains unsupported systemd characters')
    unit.write_text('[Unit]\nDescription=Panthera AWS inference loopback tunnel\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=0\n\n[Service]\nType=simple\nExecStart=/usr/bin/ssh -NT -i "'+identity+'" -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:8766:127.0.0.1:8765 '+args.user+'@'+args.host+'\nRestart=always\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','panthera-inference-tunnel.service'],check=True)
    subprocess.run(['systemctl','--user','restart','panthera-inference-tunnel.service'],check=True)
    ready = health('http://127.0.0.1:8766', result['checkpoint'], contract, 300)
    print('Ready through tunnel:', ready['checkpoint'])
    probe = subprocess.run([str(ROOT/'work/teleop-venv/bin/python'), str(ROOT/'app/remote_inference.py'), '--url', 'http://127.0.0.1:8766'], capture_output=True, text=True, timeout=90)
    if probe.returncode:
        raise RuntimeError('Synthetic inference probe failed: '+probe.stderr[-2000:]+probe.stdout[-2000:])
    print('Synthetic inference probe:', probe.stdout.strip().splitlines()[-1])
    if not args.skip_dashboard_refresh:
        refresh_dashboard(args.dashboard_url, result['checkpoint'])


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f'{Path(exc.cmd[0]).name} failed (exit {exc.returncode}); see diagnostics above.')
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        sys.exit(str(exc))
