"""Transport contract for flow-sampler guidance at stopped chunk boundaries."""
import numpy as np

RTC_PROTOCOL = 'prefix-v1'


def supports_rtc(health):
    capability = health.get('rtc', {})
    return (isinstance(capability, dict) and capability.get('supported') is True
            and capability.get('protocol') == RTC_PROTOCOL)


def prepare_rtc(previous_tail, overlap, now, max_age):
    """Align old row N with new row zero; no actions elapse while holding."""
    if previous_tail is None or not overlap:
        return None, None
    actions, observed_at = previous_tail
    if now - observed_at > max_age:
        return None, None
    count = min(overlap, len(actions))
    if not count:
        return None, None
    return dict(previous_actions=np.asarray(actions[:count]).tolist(), overlap=count), observed_at


def confirm_rtc(result, request, previous_observed_at, now, max_age):
    """Fail closed if guidance was ignored or its source observation expired."""
    if request is None:
        return 0
    confirmation = result.get('rtc')
    if (not isinstance(confirmation, dict)
            or type(confirmation.get('overlap')) is not int
            or confirmation['overlap'] != request['overlap']):
        raise RuntimeError('Inference server did not confirm requested RTC guidance')
    if now - previous_observed_at > max_age:
        raise RuntimeError('RTC previous observation expired; discard the guided prediction')
    return request['overlap']
