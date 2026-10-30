# Hardware benchmark: Apple M5 (measured 26 Oct 2026)

## Machine (verified)
- Apple M5: 10 CPU cores (4 performance + 6 efficiency), 10-core GPU, Metal 4
- 24 GB unified memory, 311 GB free disk
- macOS 26.6.2; Python 3.14.6; PyTorch 2.14.1 with MPS available

## 3D CNN training (E06)
Model: 4-block 3D CNN (16-32-64-128 channels, BatchNorm). Input 1 × 96 × 112 × 96. Forward + backward + Adam step on synthetic tensors. Median of 6 steps after 2 warm-up steps.

| Device / precision | Batch | s / step | Volumes / s | MPS memory (GB) |
|---|---|---|---|---|
| CPU fp32 | 2 | 1.124 | 1.78 | n/a |
| MPS fp32 | 2 | 0.565 | **3.54** | 2.18 |
| MPS fp32 | 4 | 1.158 | 3.45 | 3.30 |
| MPS fp32 | 8 | 2.372 | 3.37 | 5.55 |
| MPS fp16 autocast | 2 | 0.614 | 3.26 | 2.19 |
| MPS fp16 autocast | 4 | 1.229 | 3.25 | 3.30 |
| MPS fp16 autocast | 8 | 2.529 | 3.16 | 6.55 |

- **MPS is about 2× CPU**, not the large factor I expected.
- **Throughput is flat from batch 2 to 8**, so the GPU is compute-bound; bigger batches buy nothing.
- **fp16 autocast: no speedup and no memory saving.** Its outputs differ from fp32 by at most 3 × 10⁻⁶, which suggests 3D convolutions may not actually run in fp16 under MPS autocast. Not investigated further, since there was no benefit.
- Memory is not the limit at this size: batch 8 uses 5.6 GB of 24 GB.

## 2D inference (VGG16, 224 × 224)
- MPS: **126 images/s**. CPU: 31 images/s.
- Embedding the 44,000-image Kaggle set took 538 s (E02), slower than this benchmark because JPEG decoding ran on the CPU alongside other jobs.

## What this means for the project
- One epoch over about 640 ADNI-sized training volumes ≈ 3 min with this small model. 5 folds × 50 epochs ≈ 12 h. Feasible overnight; slow for iteration.
- Cost scales with voxel count, which motivates comparing compact 3D representations (hippocampal patches, regional volumes) with a full-volume CNN. That is Track E, candidate 4 in the roadmap.
- **No energy claims are made from any of this.** Energy was not measured.

## Iteration 2 measurements (27 Oct 2026, same machine)

**Frozen-backbone embedding of HAM10000** (10,015 images, 224×224, batch 64, MPS fp32, decoding excluded):

| Backbone | Parameters | Images / s |
|---|---|---|
| ResNet-18 | 11.2 M | 638 |
| ResNet-50 | 23.5 M | 195 |
| ConvNeXt-Tiny | 27.8 M | 171 |
| ViT-B/16 | 85.8 M | 61 |

- JPEG decoding and resizing of 10,015 images on the CPU took 28.8 s.
- The **whole pre-registered E10 analysis** (5 rungs × 3 fold assignments, inner-CV tuning, 2,000-resample bootstraps) took **2.4 min** on the CPU (about 4 cores busy).
- Throughput is not proportional to parameter count: ViT-B/16 has 3.1× ConvNeXt-T's parameters but runs 2.8× slower, and ResNet-50 has 2.1× ResNet-18's parameters but runs 3.3× slower.

**PatchCore-lite on VisA** (Wide-ResNet-50 layer 2+3, 1,536-d patches):
- Feature extraction for 1,100 images (900 train + 200 test, including decoding of ~1,500×1,000 JPEGs): 16.6 s, about 66 images/s end to end.
- Exact nearest-neighbour scoring of 156,800 test patches against a 70,560-patch memory bank on MPS: about 18 s per variant.

**Practical conclusion:** frozen-feature experiments on 10⁴-image datasets run in minutes on the M5, so the experiment loop is limited by thinking and downloading, not compute. End-to-end fine-tuning (the next open question for Track A) would be the first workload where compute matters. Iteration 1's 3D CNN figure (about 3.5 volumes/s) doesn't apply to these 2D workloads.
