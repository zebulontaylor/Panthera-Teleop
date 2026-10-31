#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Stopping the teleop frontend. The independent motor backend retains powered hold.'
exec systemctl --user stop panthera-gravity-assist.service
