"""E13: confirmatory replication of E10 on independent datasets.

Follows research/iteration3/PREREGISTRATION_REPLICATION.md.
Usage:  python e13_replication.py <dataset> embed|analyse
Datasets: isic2020, isic2019_nonham, pad_ufes. Embeddings are cached in
research/data/<dataset>/emb_<rung>.npy in the adapter's row order.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e10_ham_embed import BACKBONES, DEV, MEAN, STD, headless, hsv_hist, load_224

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
SEEDS = [0, 1, 2]
B = 2000


# ---------------------------------------------------------------- adapters
def find_images(root, exts=(".jpg", ".png", ".jpeg")):
    return {p.stem.replace("_downsampled", ""): p for p in root.rglob("*") if p.suffix.lower() in exts
            and not p.name.startswith("._")}


def adapter(name):
    d = RESEARCH / "data"
    if name == "isic2020":
        root = d / "isic2020/x"
        df = pd.read_csv(next(root.rglob("train.csv")))
        imgs = find_images(root / "train")
        df = df[df["image_name"].isin(imgs)].reset_index(drop=True)
        df["path"] = df["image_name"].map(imgs)
        return dict(df=df, y=df["target"].to_numpy(int), groups=df["patient_id"].to_numpy(),
                    blocks={"all": (["age_approx"], ["sex", "anatom_site_general_challenge"])},
                    task="melanoma vs rest", group_unit="patient_id")
    if name == "isic2019_nonham":
        meta = pd.read_csv(d / "isic2019/official/ISIC_2019_Training_Metadata.csv")
        gt = pd.read_csv(d / "isic2019/official/ISIC_2019_Training_GroundTruth.csv")
        df = meta.merge(gt, on="image")
        df = df[~df["lesion_id"].fillna("").str.startswith("HAM_")]
        imgs = find_images(d / "isic2019/images256/x")
        key = df["image"].str.replace("_downsampled", "", regex=False)  # file stems drop this suffix
        df = df[key.isin(imgs)].reset_index(drop=True)
        df["path"] = df["image"].str.replace("_downsampled", "", regex=False).map(imgs)
        groups = df["lesion_id"].fillna(df["image"]).to_numpy()
        return dict(df=df, y=df["MEL"].to_numpy(int), groups=groups,
                    blocks={"all": (["age_approx"], ["sex", "anatom_site_general"])},
                    task="melanoma vs rest", group_unit="lesion_id (image if missing)")
    if name == "pad_ufes":
        df = pd.read_csv(d / "pad_ufes/official/metadata.csv")
        imgs = find_images(d / "pad_ufes/kaggle/x")
        df = df[df["img_id"].str.replace(".png", "", regex=False).isin(imgs)].reset_index(drop=True)
        df["path"] = df["img_id"].str.replace(".png", "", regex=False).map(imgs)
        y = df["diagnostic"].isin(["BCC", "SCC", "MEL"]).to_numpy(int)
        D = (["age"], ["gender", "fitspatrick", "region"])
        S = ([], ["itch", "grew", "hurt", "changed", "bleed", "elevation"])
        return dict(df=df, y=y, groups=df["patient_id"].to_numpy(),
                    blocks={"all": (D[0], D[1] + S[1]), "D_demographic": D, "S_symptoms": S},
                    task="malignant (BCC/SCC/MEL) vs benign", group_unit="patient_id")
    if name in ("hiba", "msk2020", "milk10k_derm", "milk10k_clin"):
        folder = "milk10k" if name.startswith("milk10k") else name
        df = pd.read_csv(d / f"{folder}/metadata.csv")
        df = df[df["diagnosis_1"].isin(["Benign", "Malignant"])]
        if name == "hiba" or name == "milk10k_derm":
            df = df[df["image_type"] == "dermoscopic"]
        elif name == "milk10k_clin":
            df = df[df["image_type"].astype(str).str.startswith("clinical")]
        df["path"] = [d / f"{folder}/images/{i}.jpg" for i in df["isic_id"]]
        df = df[[p.exists() and p.stat().st_size > 0 for p in df["path"]]].reset_index(drop=True)
        if "lesion_id" in df:
            groups = df["lesion_id"].fillna(df["patient_id"] if "patient_id" in df else df["isic_id"])
        else:
            groups = df["isic_id"]
        if name == "hiba":
            groups = df["patient_id"].fillna(df["isic_id"])  # pre-registered: patient-level grouping
        cat = ["sex", "anatom_site_1"] + (["fitzpatrick_skin_type"] if name == "hiba" else [])
        return dict(df=df, y=(df["diagnosis_1"] == "Malignant").to_numpy(int), groups=groups.to_numpy(),
                    blocks={"all": (["age_approx"], cat)}, task="malignant vs benign",
                    group_unit="patient_id" if name == "hiba" else "lesion_id (image if missing)")
    raise ValueError(name)


def block_matrix(df, num, cat, train_idx):
    cols = []
    for c in num:
        v = pd.to_numeric(df[c], errors="coerce")
        cols += [v.fillna(v.iloc[train_idx].median()).to_numpy()[:, None] / 100, v.isna().to_numpy(float)[:, None]]
    for c in cat:
        cols.append(pd.get_dummies(df[c].astype(str).replace({"nan": "unknown", "": "unknown"}), dtype=float).to_numpy())
    return np.hstack(cols)


def oof_block(df, num, cat, folds):
    M = np.zeros((len(df), block_matrix(df, num, cat, folds[0][0]).shape[1]))
    for tr, te in folds:
        M[te] = block_matrix(df, num, cat, tr)[te]
    return M


# ---------------------------------------------------------------- embedding
@torch.no_grad()
def embed(name, chunk=1024):
    a = adapter(name)
    out_dir = RESEARCH / f"data/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)  # isic2019_nonham has no pre-existing folder
    models = {k: headless(k) for k in BACKBONES}
    feats = {k: [] for k in RUNGS}
    t0 = time.perf_counter()
    for i in range(0, len(a["df"]), chunk):
        ims = [load_224(p) for p in a["df"]["path"].iloc[i:i + chunk]]
        feats["hsv"].append(np.stack([hsv_hist(im) for im in ims]))
        x = torch.stack([torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1) for im in ims])
        x = (x - MEAN) / STD
        for k, m in models.items():
            feats[k].append(torch.cat([m(x[j:j + 64].to(DEV)).float().cpu() for j in range(0, len(x), 64)]).numpy())
        print(f"{name}: {min(i + chunk, len(a['df']))}/{len(a['df'])} ({time.perf_counter() - t0:.0f}s)", flush=True)
    for k in RUNGS:
        np.save(out_dir / f"emb_{k}.npy", np.concatenate(feats[k]).astype(np.float32))
    a["df"].drop(columns=["path"]).to_csv(out_dir / "rows.csv", index=False)
    return time.perf_counter() - t0


# ---------------------------------------------------------------- analysis
def boot_samples(d, groups, rng):
    codes, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    idx = rng.integers(0, len(codes), (B, len(codes)))
    return sums[idx].sum(1) / counts[idx].sum(1)


def recoverability_auc(X, target, mask, folds):
    p = np.full(len(target), np.nan)
    for tr, te in folds:
        tr, te = tr[mask[tr]], te[mask[te]]
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[tr]), target[tr]).decision_function(sc.transform(X[te]))
    return float(roc_auc_score(target[mask], p[mask]))


def analyse(name):
    a = adapter(name)
    df, y, groups = a["df"], a["y"], a["groups"]
    exp = Experiment(f"E13_{name}", "See research/iteration3/PREREGISTRATION_REPLICATION.md",
                     {"dataset": name, "n_images": int(len(df)), "n_groups": int(len(np.unique(groups))),
                      "positives": int(y.sum()), "task": a["task"], "group_unit": a["group_unit"]},
                     {"rungs": RUNGS, "seeds": SEEDS, "bootstrap": B}, SEEDS)
    rng = np.random.default_rng(13)
    res = {"seeds": {}}
    per_image = {}
    for seed in SEEDS:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df, y, groups))
        blocks = {b: oof_block(df, *spec, folds) for b, spec in a["blocks"].items()}
        res["seeds"][seed] = {}
        for rung in RUNGS:
            X = np.load(RESEARCH / f"data/{name}/emb_{rung}.npy")
            score, _ = oof_scores(X, y, groups, folds)
            r = {"image_auroc": round(float(roc_auc_score(y, score)), 4)}
            for b, M in blocks.items():
                l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
                d = l0 - l1
                r[f"av_{b}"] = round(float(d.mean()), 5)
                if seed == 0:
                    r[f"av_{b}_ci95"] = [round(v, 5) for v in lesion_bootstrap(d, groups, rng)]
                    per_image[(rung, b)] = d
                if b == "all":
                    r["image_logloss"] = round(float(l0.mean()), 4)
            res["seeds"][seed][rung] = r
            print(seed, rung, r, flush=True)
    # Confirmatory paired tests (seed 0), grouped bootstrap, one-sided
    tests = {}
    for hyp, (hi, lo) in {"H1": ("hsv", "convnext_tiny"), "H2": ("resnet18", "convnext_tiny")}.items():
        diff = per_image[(hi, "all")] - per_image[(lo, "all")]
        bs = boot_samples(diff, groups, rng)
        tests[hyp] = {"diff": round(float(diff.mean()), 5),
                      "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                      "p_one_sided": float((1 + np.sum(bs <= 0)) / (B + 1))}
    res["confirmatory_tests"] = tests
    if name == "pad_ufes":
        f0 = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
        ret = {}
        for b in ["D_demographic", "S_symptoms"]:
            hsv_av = per_image[("hsv", b)].mean()
            ret[b] = round(float(per_image[("convnext_tiny", b)].mean() / hsv_av), 3) if hsv_av > 0 else None
        Xc, Xh = (np.load(RESEARCH / f"data/{name}/emb_{k}.npy") for k in ("convnext_tiny", "hsv"))
        rec = {}
        for c in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
            v = df[c].astype(str)
            mask = v.isin(["True", "False"]).to_numpy()
            t = (v == "True").to_numpy(int)
            rec[c] = {"convnext": round(recoverability_auc(Xc, t, mask, f0), 3),
                      "hsv": round(recoverability_auc(Xh, t, mask, f0), 3)}
        res["mechanism"] = {"retained_fraction_hsv_to_convnext": ret, "symptom_recoverability_auroc": rec,
                            "M1_S_retains_half": ret["S_symptoms"] is not None and ret["S_symptoms"] >= 0.5,
                            "M2_D_attenuates_more": (ret["D_demographic"] is not None and ret["S_symptoms"] is not None
                                                     and ret["D_demographic"] < ret["S_symptoms"]),
                            "M3_mean_symptom_auroc_convnext": round(float(np.mean([v["convnext"] for v in rec.values()])), 3)}
    print(json.dumps({k: v for k, v in res.items() if k != "seeds"}, indent=1), flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    ds, step = sys.argv[1], sys.argv[2]
    if step == "embed":
        print("embed seconds", round(embed(ds), 1))
    else:
        analyse(ds)
