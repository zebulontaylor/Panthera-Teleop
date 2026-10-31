"""Switch workspace dashboard frontends without restarting the motor backend."""
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = ('hand_guide.py', 'inference_dashboard.py')


def frontend_pids():
    """Find this checkout's frontends, including ones launched outside systemd."""
    result = set()
    targets = {ROOT / 'app' / name for name in FRONTENDS}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if not args or not Path(os.fsdecode(args[0])).name.startswith('python'):
                continue
            cwd = (entry / 'cwd').resolve()
            for arg in args[1:]:
                path = Path(os.fsdecode(arg))
                if path.name in FRONTENDS and (cwd / path).resolve() in targets:
                    result.add(int(entry.name))
                    break
        except (OSError, RuntimeError):
            continue  # Process exited, or belongs to another user.
    return result


def stop_frontends():
    pids = frontend_pids()
    if not pids:
        return
    print('Stopping dashboard; saving recordings and retaining backend hold…', flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 35
    while frontend_pids():
        if time.monotonic() >= deadline:
            raise RuntimeError('Dashboard did not finish shutdown within 35 seconds; new dashboard was not started.')
        time.sleep(0.2)


def main():
    mode = sys.argv[1]
    extra = sys.argv[2:]
    launcher = ROOT / ('start.sh' if mode == 'teleop' else 'scripts/start-inference-dashboard.sh')
    # Help must not interrupt a running session.
    if '--help' not in extra and '-h' not in extra:
        stop_frontends()
        # Allow the exited frontend's camera devices to finish releasing.
        time.sleep(1)
        print(f'Starting {mode} dashboard on port 8093 (override with --port).', flush=True)
    os.chdir(ROOT)
    os.execv(str(launcher), [str(launcher), '--port', '8093', *extra])


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as exc:
        sys.exit(str(exc))
