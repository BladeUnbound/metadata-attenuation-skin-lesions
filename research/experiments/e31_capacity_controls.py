"""E31 (exploratory; request E21): representation quality, model size, fine-tuning, data scaling.
Registered in research/iteration4/REGISTRY_E30_E34.md before implementation.

Subcommands
  embed_family <ds>      frozen ResNet-34/101 and ConvNeXt-S/B embeddings (torchvision weights)
  analyse_family         within-family size ladders: ResNet-18/34/50/101, ConvNeXt-T/S/B
  scaling_frozen         image head trained on 10/25/50/100% of training groups (test folds fixed),
                         frozen ResNet-18 and ConvNeXt-T; 2 subsample repeats; HAM10000, ISIC 2019
  ft_fraction            fine-tuned ResNet-18 on HAM10000 at 10/25/50% (100% = E14), checkpointed
Design note: the training fraction applies to the image model only; the metadata second
stage uses the standard cross-fitting over all folds, so differences reflect image quality.
Per condition: image AUROC, log-loss, Brier, calibration, metadata added value with
grouped-bootstrap CI (complete cases, seed-0 folds). Paired differences within a family.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import fit_tuned
from e10_ham_embed import DEV, MEAN, STD, load_224
from x_common import (VARS, available, block, ci, emb, folds_for, full_metrics, group_boot_mean, oof_probs,
                      per_image_ll, second_stage, subset)
from sklearn.preprocessing import StandardScaler

FAMILY_NEW = {"resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1),
              "resnet101": (tvm.resnet101, tvm.ResNet101_Weights.IMAGENET1K_V2),
              "convnext_small": (tvm.convnext_small, tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1),
              "convnext_base": (tvm.convnext_base, tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)}
LADDERS = {"resnet": ["resnet18", "resnet34", "resnet50", "resnet101"],
           "convnext": ["convnext_tiny", "convnext_small", "convnext_base"]}
FAMILY_DATASETS = ["ham10000", "pad_ufes", "isic2019_nonham"]
B = 1000


@torch.no_grad()
def embed_family(ds, chunk=512):
    from e18_foundation_rung import load_dataset
    d = load_dataset(ds)
    for name, (fn, w) in FAMILY_NEW.items():
        out = d["dir"] / f"emb_{name}.npy"
        if out.exists():
            continue
        m = fn(weights=w)
        if name.startswith("resnet"):
            m.fc = torch.nn.Identity()
        else:
            m.classifier[2] = torch.nn.Identity()
        m = m.eval().to(DEV)
        feats, t0 = [], time.perf_counter()
        for i in range(0, len(d["paths"]), chunk):
            x = torch.stack([torch.from_numpy(np.asarray(load_224(p), np.float32) / 255).permute(2, 0, 1)
                             for p in d["paths"][i:i + chunk]])
            x = (x - MEAN) / STD
            feats.append(torch.cat([m(x[j:j + 64].to(DEV)).float().cpu() for j in range(0, len(x), 64)]).numpy())
        np.save(out, np.concatenate(feats).astype(np.float32))
        print(f"{ds} {name}: {len(d['paths'])} images, {time.perf_counter() - t0:.0f}s", flush=True)
        del m
        if DEV == "mps":
            torch.mps.empty_cache()


def condition(s, score, M, folds, rng):
    y, g = s["y"], s["g"]
    l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
    d = l0 - l1
    return {"image": full_metrics(y, oof_probs(score[:, None], y, folds)),
            "delta": float(d.mean()), "delta_ci95": ci(group_boot_mean(d, g, rng, B)), "_d": d}


def scores_with_fraction(X, s, folds, frac, rng):
    """Out-of-fold image scores where each fold's head sees only `frac` of its training groups."""
    y, g = s["y"], s["g"]
    score = np.zeros(len(y))
    for tr, te in folds:
        units = np.unique(g[tr])
        pick = set(rng.choice(units, max(2, int(round(frac * len(units)))), replace=False)) if frac < 1 else set(units)
        sub = tr[np.isin(g[tr], list(pick))]
        if y[sub].min() == y[sub].max():
            sub = tr
        sc = StandardScaler().fit(X[sub])
        head, _ = fit_tuned(sc.transform(X[sub]), y[sub], g[sub])
        score[te] = head.decision_function(sc.transform(X[te]))
    return score


def analyse_family():
    rng = np.random.default_rng(311)
    exp = Experiment("E31a_family_size", "H31: within a family, metadata added value falls as image log-loss improves.",
                     {"datasets": FAMILY_DATASETS, "rows": "complete cases"}, {"ladders": LADDERS}, 0)
    from e10_ham_attenuation import oof_scores
    res = {}
    for ds in FAMILY_DATASETS:
        if not available(ds):
            continue
        s = subset(ds, True)
        folds = folds_for(s)
        M = block(s, VARS[ds], folds)
        res[ds] = {}
        for fam, ladder in LADDERS.items():
            per = {}
            for rung in ladder:
                X = emb(s, rung)
                if X is None:
                    continue
                score, _ = oof_scores(X, s["y"], s["g"], folds)
                c = condition(s, score, M, folds, rng)
                per[rung] = c["_d"]
                res[ds][rung] = {k: v for k, v in c.items() if k != "_d"}
                print(ds, rung, round(c["image"]["auroc"], 4), round(1000 * c["delta"], 2), flush=True)
            if len(per) >= 2:
                lo, hi = ladder[0], [r for r in ladder if r in per][-1]
                diff = per[lo] - per[hi]
                res[ds][f"{fam}:{lo}-{hi}"] = {"diff": float(diff.mean()), "ci95": ci(group_boot_mean(diff, s["g"], rng, B))}
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


def scaling_frozen():
    rng = np.random.default_rng(312)
    fracs, reps = [0.1, 0.25, 0.5, 1.0], 2
    exp = Experiment("E31b_scaling_frozen", "H31: metadata added value is larger when the image model sees less data.",
                     {"datasets": ["ham10000", "isic2019_nonham"], "rows": "complete cases"},
                     {"fractions": fracs, "repeats": reps, "rungs": ["resnet18", "convnext_tiny"]}, 0)
    res = {}
    for ds in ["ham10000", "isic2019_nonham"]:
        s = subset(ds, True)
        folds = folds_for(s)
        M = block(s, VARS[ds], folds)
        res[ds] = {}
        for rung in ["resnet18", "convnext_tiny"]:
            X = emb(s, rung)
            for f in fracs:
                for r in range(reps if f < 1 else 1):
                    score = scores_with_fraction(X, s, folds, f, rng)
                    c = condition(s, score, M, folds, rng)
                    res[ds][f"{rung}|{f}|{r}"] = {k: v for k, v in c.items() if k != "_d"}
                    print(ds, rung, f, r, round(c["image"]["auroc"], 4), round(1000 * c["delta"], 2), flush=True)
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


def ft_fraction():
    from e14_finetune import load_256, train_fold
    from e18_foundation_rung import load_dataset
    rng = np.random.default_rng(313)
    d = load_dataset("ham10000")
    s = subset("ham10000", True)
    folds = folds_for(s)
    M = block(s, VARS["ham10000"], folds)
    imgs_all = [load_256(p) for p in d["paths"]]
    imgs = [imgs_all[i] for i in s["idx"]]
    ck = RESEARCH / "data/ham10000/e31c"
    ck.mkdir(exist_ok=True)
    exp = Experiment("E31c_ft_fraction", "H31: fine-tuned image models trained on less data leave more metadata value.",
                     {"dataset": "ham10000 (complete cases)"}, {"fractions": [0.1, 0.25, 0.5], "schedule": "as E14"}, 0)
    res = {}
    for f in [0.1, 0.25, 0.5]:
        score = np.zeros(len(s["y"]))
        for k, (tr, te) in enumerate(folds):
            p = ck / f"r18_f{f}_fold{k}.npy"
            if p.exists():
                score[te] = np.load(p)
                continue
            units = np.unique(s["g"][tr])
            pick = rng.choice(units, int(round(f * len(units))), replace=False)
            sub = tr[np.isin(s["g"][tr], pick)]
            score[te] = train_fold("resnet18", imgs, s["y"], sub, te, seed=k)
            np.save(p, score[te])
            print(f"fraction {f} fold {k} done", flush=True)
        c = condition(s, score, M, folds, rng)
        res[str(f)] = {k: v for k, v in c.items() if k != "_d"}
        print("fraction", f, round(c["image"]["auroc"], 4), round(1000 * c["delta"], 2), flush=True)
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


def ft_fraction_ext():
    """E31d (addendum after review round 5): 100% on complete cases, plus a second subsample at 10/25/50%."""
    from e14_finetune import load_256, train_fold
    from e18_foundation_rung import load_dataset
    from scipy.stats import spearmanr
    d = load_dataset("ham10000")
    s = subset("ham10000", True)
    folds = folds_for(s)
    M = block(s, VARS["ham10000"], folds)
    imgs_all = [load_256(p) for p in d["paths"]]
    imgs = [imgs_all[i] for i in s["idx"]]
    ck = RESEARCH / "data/ham10000/e31c"
    exp = Experiment("E31d_ft_fraction_ext", "Added value decreases with the fine-tuning training fraction (from E31c; exploratory).",
                     {"dataset": "ham10000 (complete cases)"},
                     {"runs": "f=1.0 rep 0; f in 0.1/0.25/0.5 rep 1 (subsample seed 314); rep 0 of f<1 reused from E31c",
                      "schedule": "as E14"}, 0)
    rngs = {1: np.random.default_rng(314)}
    res, pts, rng_b = {}, [], np.random.default_rng(315)
    for f, rep in [(1.0, 0), (0.1, 0), (0.25, 0), (0.5, 0), (0.1, 1), (0.25, 1), (0.5, 1)]:
        score = np.zeros(len(s["y"]))
        for k, (tr, te) in enumerate(folds):
            p = ck / (f"r18_f{f}_fold{k}.npy" if rep == 0 else f"r18_f{f}_rep{rep}_fold{k}.npy")
            if p.exists():
                score[te] = np.load(p)
                continue
            if f < 1:
                assert rep == 1, "repeat-0 checkpoints must come from E31c"
                units = np.unique(s["g"][tr])
                sub = tr[np.isin(s["g"][tr], rngs[1].choice(units, int(round(f * len(units))), replace=False))]
            else:
                sub = tr
            score[te] = train_fold("resnet18", imgs, s["y"], sub, te, seed=k)
            np.save(p, score[te])
            print(f"fraction {f} rep {rep} fold {k} done", flush=True)
        c = condition(s, score, M, folds, rng_b)
        res[f"{f}|{rep}"] = {k: v for k, v in c.items() if k != "_d"}
        pts.append((f, c["delta"]))
        print("fraction", f, "rep", rep, round(c["image"]["auroc"], 4), round(1000 * c["delta"], 2), flush=True)
    res["mean_by_fraction"] = {str(f): float(np.mean([v for g, v in pts if g == f])) for f in (0.1, 0.25, 0.5, 1.0)}
    res["spearman_fraction_vs_delta"] = float(spearmanr([p[0] for p in pts], [p[1] for p in pts])[0])
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"embed_family": lambda: embed_family(sys.argv[2]), "analyse_family": analyse_family,
     "scaling_frozen": scaling_frozen, "ft_fraction": ft_fraction, "ft_fraction_ext": ft_fraction_ext}[cmd]()
