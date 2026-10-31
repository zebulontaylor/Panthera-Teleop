---
pretty_name: Panthera pin insertion failure clips (October 27, 2026)
task_categories:
- robotics
---

# Panthera pin insertion failure clips

A 30.4-second video isolating pin-insertion failures from the two latest full HITL attempts on October 27, 2026. The insertion wrist is shown large, with opposite wrist and overhead views alongside it.

[Watch or download the MP4](https://huggingface.co/datasets/ZebTaylor/panthera-pin-insertion-failure-clips-20261027/resolve/main/pin-insertion-failures-two-latest-attempts.mp4)

<video controls width="100%" src="https://huggingface.co/datasets/ZebTaylor/panthera-pin-insertion-failure-clips-20261027/resolve/main/pin-insertion-failures-two-latest-attempts.mp4"></video>

- 0:00–0:20.6: attempt started 10:54:47 AM CDT, source `20261027T155447-6d9ceba2`, insertion approach through unsuccessful end (recording segments 19–42).
- 0:20.6–0:30.4: attempt started 10:59:06 AM CDT, source `20261027T155906-04975fdb`, final insertion approach through unsuccessful end (recording segments 13–19).

Setup, earlier task stages, human correction, and held periods are omitted. Observed motion speed is preserved using host timestamps within each active segment. One-second frozen frames introduce and end each attempt. Camera views use approximately synchronized host receive timestamps; cameras are not hardware synchronized. Both source recordings are marked unsuccessful. A 0.128-second intervening gripper-feedback fault recording is excluded because it contains no insertion attempt.

`clip-manifest.json` records the exact source segments and time ranges. No raw robot state or unrelated recordings are included.
