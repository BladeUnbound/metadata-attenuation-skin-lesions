# Paused, 26 Oct 2026 16:18 IST, at your request

Nothing is running. Partial downloads are kept, and curl resumes them from where they stopped (`-C -`).

| File | Downloaded | Full size |
|---|---|---|
| `data/ham10000/raw/part1.zip` | 0.81 GB | 1.37 GB |
| `data/ham10000/raw/part2.zip` | 0.83 GB | 1.40 GB |
| `data/visa/VisA_20220922.tar` | 0.80 GB | 1.93 GB |
| HAM10000 metadata, masks, ISIC 2018 test set | complete | n/a |

## On "continue", in this order
1. Resume HAM10000 part 1 and part 2 (Dataverse file IDs 3172585, 3172584) with `curl -C -`, then extract to `data/ham10000/images`, `masks` and `test_images`.
2. `novelty/.venv/bin/python -I research/experiments/e10_ham_embed.py` (all 5 rungs, about 3 min).
3. `novelty/.venv/bin/python -I research/experiments/e10_ham_attenuation.py` (full E10, about 5 min). Add the exploratory leave-one-source-out control.
4. `novelty/.venv/bin/python -I research/experiments/e12_ham_cam_lesion.py`
5. Resume VisA (S3 URL in `iteration2/DATASETS.md`) with `curl -C -`, extract, then run `e11_visa_anisotropy.py`.
6. Update EXPERIMENTS.md, NOVELTY_LEDGER.md, ROADMAP.md and PUBLICATION_STATUS.md.

The dev-run outputs (`emb_dev_*.npy`, `meta_dev.csv`, `results/E10_ham_attenuation_DEV.json`) are pipeline checks only and must not be reported as results.
