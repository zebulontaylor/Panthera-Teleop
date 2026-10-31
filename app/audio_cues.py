"""Short local tones played off the recording and motor threads."""
import math
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import wave


# Keep collection sounds stable; inference transitions have their own motifs.
PATTERNS = dict(recording_start=[740, 988, 1480],
                end=[660, 990], press=[880], kept=[1100],
                review=[700, 700], discarded=[440, 330, 220], cancel=[220],
                policy_play=[660, 880, 1100], policy_pause=[880, 660],
                intervention_start=[440, 880], intervention_end=[880, 440],
                handoff=[660, 990, 660])


class AudioCues:
    def __init__(self, enabled=True, report=lambda **values: None):
        self.report = report
        self.player = next((p for p in ('pw-play', 'paplay', 'aplay') if shutil.which(p)), None) if enabled else None
        self.events = queue.Queue(maxsize=32)
        self.thread = None
        if enabled and not self.player:
            report(audio_error='Audio cues unavailable: install pw-play, paplay, or aplay')
        if self.player:
            self.thread = threading.Thread(target=self.worker, name='episode-audio', daemon=True)
            self.thread.start()

    def play(self, cue):
        if self.thread:
            if cue not in PATTERNS:
                self.report(audio_error=f'Unknown audio cue: {cue}')
                return
            try:
                self.events.put_nowait(cue)
            except queue.Full:
                self.report(audio_error='Audio cue queue full')

    def worker(self):
        with tempfile.TemporaryDirectory(prefix='panthera-audio-') as directory:
            paths = {}
            for cue, frequencies in PATTERNS.items():
                path = Path(directory) / (cue + '.wav')
                samples = bytearray()
                rate = 24000
                for frequency in frequencies:
                    length = int(rate * 0.075)
                    for i in range(length):
                        envelope = min(1, i / 120, (length - i) / 120)
                        samples.extend(struct.pack('<h', int(7000 * envelope * math.sin(2 * math.pi * frequency * i / rate))))
                    samples.extend(b'\0\0' * int(rate * 0.025))
                with wave.open(str(path), 'wb') as output:
                    output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                    output.writeframes(samples)
                paths[cue] = path
            while True:
                cue = self.events.get()
                if cue is None:
                    return
                try:
                    subprocess.run([self.player, str(paths[cue])], check=True, timeout=2,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except (OSError, subprocess.SubprocessError) as exc:
                    self.report(audio_error=f'Audio cue playback failed: {exc}')

    def close(self):
        if self.thread and self.thread.is_alive():
            # Discard stale cues at shutdown.
            while True:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    break
            self.events.put_nowait(None)
            self.thread.join(timeout=3)
