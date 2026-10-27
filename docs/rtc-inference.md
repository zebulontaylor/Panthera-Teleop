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
requested overlap. The dashboard rejects missing acknowledgements and expired
current observations before executing a new chunk. The old plan is a continuity
prior, so its source timestamp is not subject to the current sensor deadline;
that age includes previous inference and safety-retimed playback. Non-play
commands, pause, intervention, and faults invalidate the prior. A guidance error
stops playback and holds instead of retrying with independent chunks.

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

## Deployment verification

Deployed October 27, 2026 to the existing `mixed70-real30-008500` service. Its
0.16 CUDA allocator cap and checkpoint deployment checks were retained. The
inference-only synthetic probe acknowledged 10 guided actions on three requests
at 0.40–0.43 seconds each. The first unguided warmup took 3.79 seconds; subsequent
guided requests were within the unchanged two-second request-age limit. This
checks the real GPU sampler, not physical task success.

See `inference-rtc-deployment-verification.json` for the full result, file hashes
and backup location. The dashboard was restarted and left disconnected; the
separate motor backend confirmed powered hold. Reconnect from the dashboard when
ready for its normal collection-pose reset.

For server rollback, restore `server.py` from
`/mnt/pi05/inference/backups/pre-rtc-20261027T225940Z/` and restart only
`panthera-inference.service` on the inference host. The original checkpoint,
contract, memory cap and service configuration are in that backup. The extra RTC
helper is inert with the old server. Reconnect the dashboard to negotiate the
restored server's capabilities; it will select crossfading again.
