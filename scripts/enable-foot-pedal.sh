#!/usr/bin/env bash
# Grant the local desktop user access only to this PCsensor USB foot switch.
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
    echo 'Run with sudo: sudo ./scripts/enable-foot-pedal.sh' >&2
    exit 1
fi
# uaccess must be tagged before systemd's 73-seat-late.rules applies ACLs.
cat > /etc/udev/rules.d/70-panthera-foot-pedal.rules <<'RULE'
SUBSYSTEM=="input", KERNEL=="event*", ATTRS{idVendor}=="3553", ATTRS{idProduct}=="b001", TAG+="uaccess"
RULE
rm -f /etc/udev/rules.d/99-panthera-foot-pedal.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=input --action=add
udevadm settle
echo 'PCsensor foot pedal access enabled. Restart recording to capture the pedal.'
