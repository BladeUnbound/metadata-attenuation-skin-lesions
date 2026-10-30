"""E06: What does the M5 actually deliver for 3D MRI training, and where are the limits?

Workloads (synthetic tensors; timing and memory only, no accuracy claims):
  A. small 3D CNN, input 1 x 96 x 112 x 96 (a typical downsampled T1 volume),
     forward + backward + Adam step, batch sizes 2 / 4 / 8
     devices: CPU fp32, MPS fp32, MPS fp16 autocast
  B. VGG16 2D inference at 224 x 224 (re-evaluating the original model)
Also checks fp16 numerical agreement against fp32 on identical weights and inputs.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from vgg_port import KerasVGGHead

SEED = 0
VOL = (96, 112, 96)


def block(c_in, c_out):
    return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm3d(c_out),
                         nn.ReLU(inplace=True), nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
                         nn.BatchNorm3d(c_out), nn.ReLU(inplace=True), nn.MaxPool3d(2))


def small_3d_cnn():
    return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, 1))


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def bench_train(dev, batch, fp16, steps=6, warmup=2):
    torch.manual_seed(SEED)
    model = small_3d_cnn().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    x = torch.randn(batch, 1, *VOL, device=dev)
    y = torch.randint(0, 2, (batch, 1), device=dev).float()
    if dev == "mps":
        torch.mps.empty_cache()
    times = []
    for i in range(warmup + steps):
        sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev, dtype=torch.float16, enabled=fp16):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync(dev)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    mem = torch.mps.driver_allocated_memory() / 2**30 if dev == "mps" else None
    t = sorted(times)[len(times) // 2]
    return {"median_step_s": round(t, 4), "volumes_per_s": round(batch / t, 2),
            "mps_driver_mem_gb": round(mem, 2) if mem else None}


def fp16_agreement():
    torch.manual_seed(SEED)
    model = small_3d_cnn().to("mps").eval()
    x = torch.randn(4, 1, *VOL, device="mps")
    with torch.no_grad():
        ref = model(x)
        with torch.autocast("mps", dtype=torch.float16):
            half = model(x).float()
    return {"max_abs_logit_diff_fp16_vs_fp32": float((half - ref).abs().max()),
            "logit_scale": float(ref.abs().max())}


def bench_vgg(dev, batch=32, reps=5):
    model = KerasVGGHead().to(dev).eval()
    x = torch.rand(batch, 3, 224, 224, device=dev)
    with torch.no_grad():
        model(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        sync(dev)
    return round(batch * reps / (time.perf_counter() - t0), 1)


def main():
    exp = Experiment(
        "E06_m5_benchmark",
        "MPS gives a large speedup over CPU for 3D CNN training; fp16 autocast helps further without "
        "harming numerical agreement; memory limits batch size before compute does.",
        {"source": "synthetic tensors"},
        {"volume": VOL, "model": "4-block 3D CNN (16-32-64-128), BatchNorm", "torch": torch.__version__},
        SEED,
    )
    res = {"train_3d": {}}
    for dev, fp16, batches in (("cpu", False, [2]), ("mps", False, [2, 4, 8]), ("mps", True, [2, 4, 8])):
        for b in batches:
            key = f"{dev}_{'fp16' if fp16 else 'fp32'}_b{b}"
            try:
                res["train_3d"][key] = bench_train(dev, b, fp16)
            except Exception as e:  # out-of-memory or unsupported op: record, don't hide
                res["train_3d"][key] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(key, res["train_3d"][key], flush=True)
    res["fp16_check"] = fp16_agreement()
    res["vgg16_inference_img_per_s"] = {"cpu": bench_vgg("cpu", batch=8, reps=2), "mps": bench_vgg("mps")}
    print(res["fp16_check"], res["vgg16_inference_img_per_s"], flush=True)
    exp.finish(res, interpretation="See research/HARDWARE.md.")


if __name__ == "__main__":
    main()
