"""Transport contract for flow-sampler guidance at stopped chunk boundaries."""
import numpy as np

RTC_PROTOCOL = 'prefix-v1'


def supports_rtc(health):
    capability = health.get('rtc', {})
    return (isinstance(capability, dict) and capability.get('supported') is True
            and capability.get('protocol') == RTC_PROTOCOL)


def prepare_rtc(previous_tail, overlap):
    """Align old row N with new row zero at a stopped playback boundary.

    The tail is a continuity prior, not a sensor observation. Its source age
    includes inference and safety-retimed execution, so the current-observation
    deadline does not apply. The controller clears it on playback discontinuities.
    """
    if previous_tail is None or not overlap:
        return None
    actions, _observed_at = previous_tail
    count = min(overlap, len(actions))
    if not count:
        return None
    return dict(previous_actions=np.asarray(actions[:count]).tolist(), overlap=count)


def confirm_rtc(result, request):
    """Fail closed if the server ignored or changed the requested guidance."""
    if request is None:
        return 0
    confirmation = result.get('rtc')
    if (not isinstance(confirmation, dict)
            or type(confirmation.get('overlap')) is not int
            or confirmation['overlap'] != request['overlap']):
        raise RuntimeError('Inference server did not confirm requested RTC guidance')
    return request['overlap']
