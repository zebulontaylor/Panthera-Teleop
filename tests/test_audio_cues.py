"""Audio cues use valid WAVs and report failures without blocking recording."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from audio_cues import AudioCues


class AudioCueTests(unittest.TestCase):
    def test_start_and_inference_cues_have_valid_distinct_wavs(self):
        names = ('recording_start', 'policy_play', 'policy_pause', 'intervention_start',
                 'intervention_end', 'handoff')
        played = threading.Event()
        recordings = {}

        def run(command, **kwargs):
            self.assertNotEqual(threading.current_thread(), threading.main_thread())
            with wave.open(command[1]) as cue:
                self.assertEqual(cue.getnchannels(), 1)
                self.assertEqual(cue.getsampwidth(), 2)
                self.assertEqual(cue.getframerate(), 24000)
                self.assertGreater(cue.getnframes(), 0)
                recordings[Path(command[1]).stem] = cue.readframes(cue.getnframes())
            if len(recordings) == len(names):
                played.set()

        with patch('audio_cues.shutil.which', return_value='/usr/bin/pw-play'), \
             patch('audio_cues.subprocess.run', side_effect=run):
            audio = AudioCues()
            try:
                for name in names:
                    audio.play(name)
                self.assertTrue(played.wait(2))
                self.assertEqual(set(recordings), set(names))
                self.assertEqual(len(set(recordings.values())), len(names))
            finally:
                audio.close()

    def test_play_does_not_wait_for_playback_and_unknown_cue_is_reported(self):
        started = threading.Event()
        release = threading.Event()
        report = Mock()

        def run(command, **kwargs):
            started.set()
            release.wait(2)

        with patch('audio_cues.shutil.which', return_value='/usr/bin/pw-play'), \
             patch('audio_cues.subprocess.run', side_effect=run):
            audio = AudioCues(report=report)
            try:
                audio.play('unknown')
                report.assert_called_once_with(audio_error='Unknown audio cue: unknown')
                audio.play('policy_play')
                self.assertTrue(started.wait(2))
                audio.play('policy_pause')
                self.assertFalse(release.is_set())
                self.assertTrue(audio.thread.is_alive())
            finally:
                release.set()
                audio.close()

    def test_playback_uses_valid_wav_on_background_thread(self):
        played = threading.Event()
        def run(command, **kwargs):
            self.assertNotEqual(threading.current_thread(), threading.main_thread())
            self.assertEqual(command[0], 'pw-play')
            with wave.open(command[1]) as cue:
                self.assertEqual(cue.getnchannels(), 1)
                self.assertGreater(cue.getnframes(), 0)
            played.set()
        with patch('audio_cues.shutil.which', return_value='/usr/bin/pw-play'), \
             patch('audio_cues.subprocess.run', side_effect=run):
            audio = AudioCues()
            try:
                audio.play('end')
                self.assertTrue(played.wait(2))
            finally:
                audio.close()

    def test_disabled_and_missing_player(self):
        report = Mock()
        with patch('audio_cues.shutil.which', return_value=None):
            disabled = AudioCues(False, report)
            disabled.play('press')
            disabled.close()
            report.assert_not_called()
            missing = AudioCues(True, report)
            report.assert_called_once()
            missing.close()

    def test_playback_failure_is_reported(self):
        failed = threading.Event()
        with patch('audio_cues.shutil.which', return_value='/usr/bin/pw-play'), \
             patch('audio_cues.subprocess.run', side_effect=OSError('no audio output')):
            audio = AudioCues(report=lambda **values: failed.set())
            try:
                audio.play('press')
                self.assertTrue(failed.wait(2))
            finally:
                audio.close()
