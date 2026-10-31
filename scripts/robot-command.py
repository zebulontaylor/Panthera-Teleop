#!/usr/bin/env python3
"""Send one supported intent to the running local hand-guidance backend."""
import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='Show commands and supported phrases')
    action.add_argument('--command', help='Structured command name')
    action.add_argument('--instruction', help='A supported plain-English instruction')
    parser.add_argument('--parameters', default='{}', help='JSON object with explicit command parameters')
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    try:
        if args.list:
            with urlopen(base + '/api/commands', timeout=5) as response:
                result = json.load(response)
        else:
            parameters = json.loads(args.parameters)
            if not isinstance(parameters, dict):
                raise ValueError('--parameters must be a JSON object')
            payload = dict(parameters=parameters)
            payload['command' if args.command else 'instruction'] = args.command or args.instruction
            with urlopen(base + '/api/status', timeout=5) as response:
                token = json.load(response)['token']
            request = Request(base + '/api/commands', data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json', 'X-Session-Token': token})
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
        print(json.dumps(result, indent=2))
        return 0
    except HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        return 1
    except (URLError, ValueError, KeyError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
