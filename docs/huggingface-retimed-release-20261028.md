# Published retimed Panthera datasets

Both new releases are public and independently verified. Existing datasets, checkpoints and training configuration were not changed.

| Release | Episodes | Frames | Pinned revision |
| --- | ---: | ---: | --- |
| [real](https://huggingface.co/datasets/ZebTaylor/panthera-gear-carrier-pin-retimed-real-20hz) | 255 | 87,444 | `cae3a648340c24f45a6dabe436844d3ade06c97f` |
| [restyled](https://huggingface.co/datasets/ZebTaylor/panthera-gear-carrier-pin-retimed-restyled-20hz) | 400 | 129,892 | `83162f9578196d832a3335a37dcee8efb3e063f4` |

Both repositories expose a verified `v3.0` tag and explicit viewer patterns selecting only `data/chunk-*/*.parquet`. Use the pinned commit above for reproducible training.

The real release combines the prior 227 retimed episodes (76,273 frames) with all 28 new reference recordings at original timing (11,171 frames). The restyled release contains both existing variants of 200 prior originals, with exactly the same saved source timing maps and numeric trajectories as their paired real versions. The 28 new episodes remain real-only.

Real verification covers all 87,444 rows, 262,332 unchanged JPEG cells, 765 native 50-step action chunks, numeric/image statistics, protected timing, precision labels, and original source timestamps. Restyled verification covers all 129,892 paired numeric rows, 389,676 unchanged JPEG cells, 800 native 50-step chunks, and the 84 pinned source files before/after export. Uploaded file contents were verified through remote SHA256/LFS hashes; public visibility and tagged metadata counts were checked without authentication. All 36 scoped tests passed.

Real source: `data/processed/panthera-retimed-real255-20hz`. Restyled source on the existing compute host: `/mnt/pi05/retiming-release-20261028/restyled400`. Local publication receipts and per-repository file counts are preserved in `docs/huggingface-retimed-release-20261028.json`. Retiming metrics and limitations remain in `docs/training-retiming-export.md`. Group any future evaluation split by original source identity across real/styled versions.
