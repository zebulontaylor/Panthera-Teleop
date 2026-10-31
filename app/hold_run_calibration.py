"""Guided endpoint capture while a separate leased collector executes a sequence."""
from guided_hand_calibration import GuidedHandCalibration


class HoldRunCalibration(GuidedHandCalibration):
    name = 'hold'

    def command(self, action, settings=None):
        if action in ('hold', 'release'):
            raise ValueError('hold_and_release_are_transport_actions')
        return super().command(action, settings)

    def status(self):
        return dict(super().status(), requires_sequence_collector=True,
            movement_note='Requires the separate sequence collector and a measured motion gate. Release stops position commands.')
