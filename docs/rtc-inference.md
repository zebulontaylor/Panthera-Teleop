# Flow-sampler overlap guidance

The inference server can guide pi0.5's flow sampler with the unexecuted tail of
the preceding prediction. This replaces the dashboard's post-generation
crossfade when the server advertises `rtc.supported: true` and protocol
`prefix-v1` in `/health`. Older servers continue to use the existing crossfade;
the dashboard labels the active method.

This implementation retains stopped step boundaries. The robot holds during
inference, so old action N aligns with new action zero after executing N actions,
and `inference_delay` is zero. It does not hide inference latency or remove the
trajectory's stop at each boundary. Asynchronous execution would require an
additional controller scheduling change that accounts for safety retiming.

The existing overlap setting controls the number of previous actions used for
sampler guidance; zero disables it. A full 50-action execution leaves no tail.
The first prediction and single Execute/Preview requests have no prefix. Pause,
takeover, faults and reconnect discard the old plan. EXPO collection remains
unguided. Existing position validation, speed/acceleration bounds, tracking
checks and powered hold behavior still apply.

Prefixes are absolute physical actions on the wire. The server uses checkpoint
**action** normalization before padding them for the model, and guidance only
constrains the 14 physical action dimensions. Guidance changes denoising itself;
there is no second client crossfade. Each guided response must acknowledge the
requested overlap. The dashboard rejects missing acknowledgements and stale
source observations before executing a new chunk. A guidance error stops playback
and holds instead of retrying with independent chunks.

Deploy the inference server and `rtc_guidance.py` together using the deployment
script with the correct `PANTHERA_SSH_HOST`. Reconnect the dashboard to refresh
capabilities after a server update. The label **RTC overlap actions** confirms
capability negotiation; a `play_blend` log record with `method: "rtc"` and a
positive `blended_actions` count confirms that a guided response was accepted.
No hardware execution is required to run the automated tests.

RTC guidance needs denoiser input gradients and can increase GPU memory use and
latency. Local CPU tests cover normalization, the guidance calculation and client
handoff; deployed GPU latency and physical task success need validation on the
actual checkpoint. Existing observation-age limits are unchanged, so an overly
slow guided request is rejected.
