"""E01: Is the Keras -> PyTorch weight transfer correct, and how wrong was the notebook's?

Reference: Keras 3 (PyTorch backend) loading alzheimer_cnn.h5 directly.
Candidates:  (a) vgg_port.load_from_h5, the faithful port
             (b) the notebook's cell-18 mapping, reproduced line for line
Inputs: real MRI slices from the Kaggle dataset if present, else random images.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
import torch
from PIL import Image
from torchvision.models import vgg16

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT, RESEARCH, Experiment
from vgg_port import load_from_h5

SEED, N = 0, 64
H5 = PROJECT / "alzheimer_cnn.h5"
DATA = RESEARCH / "data/kaggle_alz/extracted"


def load_inputs(rng):
    files = sorted(DATA.rglob("*.jpg")) if DATA.exists() else []
    if files:
        pick = [files[i] for i in rng.choice(len(files), N, replace=False)]
        x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)) for p in pick])
        return x.astype(np.float32) / 255.0, f"{N} random Kaggle images (seed {SEED})"
    return rng.random((N, 224, 224, 3), dtype=np.float32), f"{N} uniform-noise images (dataset not downloaded)"


def notebook_cell18_model(keras_weights):
    """The notebook's mapping, reproduced as written (including its bugs)."""
    torch.manual_seed(SEED)
    m = vgg16(weights=None)
    m.classifier[6] = torch.nn.Linear(4096, 4)
    state = m.state_dict()
    conv_idx = fc_idx = 0
    copied = []
    for name, param in state.items():
        if "features" in name and "weight" in name:
            kw = keras_weights[conv_idx]
            if kw.ndim == 4:
                param.copy_(torch.from_numpy(np.transpose(kw, (3, 2, 0, 1))))
                copied.append((name, conv_idx))
                conv_idx += 2
        elif "classifier" in name and "weight" in name:
            kw = keras_weights[-(4 * 2 - fc_idx * 2)]
            if kw.ndim == 2:
                param.copy_(torch.from_numpy(kw.T))
                copied.append((name, -(4 * 2 - fc_idx * 2)))
                fc_idx += 1
    m.load_state_dict(state)
    return m.eval(), copied


def main():
    rng = np.random.default_rng(SEED)
    exp = Experiment(
        "E01_weight_transfer",
        "A faithful port reproduces the Keras model's outputs; the notebook's cell-18 mapping does not.",
        {"model": "alzheimer_cnn.h5 (sha256 in research/ORIGINALS.sha256)"},
        {"n_inputs": N, "keras": keras.__version__, "torch": torch.__version__},
        SEED,
    )
    keras_model = keras.models.load_model(H5, compile=False)
    x, provenance = load_inputs(rng)
    exp.record["data"]["inputs"] = provenance

    ref = np.asarray(keras_model.predict(x, verbose=0, batch_size=16), dtype=np.float64)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2)
    with torch.no_grad():
        port = load_from_h5(H5)(xt).double().numpy()
        old_model, copied = notebook_cell18_model(keras_model.get_weights())
        old = torch.softmax(old_model(xt), 1).double().numpy()

    def compare(p):
        return {
            "max_abs_prob_diff": float(np.abs(p - ref).max()),
            "argmax_agreement": float((p.argmax(1) == ref.argmax(1)).mean()),
        }

    results = {
        "keras_pred_class_counts": np.bincount(ref.argmax(1), minlength=4).tolist(),
        "faithful_port": compare(port),
        "notebook_cell18": compare(old),
        "notebook_cell18_tensors_copied": len(copied),
        "notebook_cell18_classifier_tensors_copied": sum("classifier" in n for n, _ in copied),
    }
    ok = results["faithful_port"]["argmax_agreement"] == 1.0 and results["faithful_port"]["max_abs_prob_diff"] < 2e-2
    exp.finish(
        results,
        interpretation=(
            "Faithful port matches Keras (difference is float16 vs float32 compute). "
            if ok else "Faithful port does NOT match Keras: investigate before any SNN work. "
        ) + "Notebook mapping copies only the 13 conv kernels; no biases and no classifier weights.",
        next_step="Use vgg_port for any re-evaluation or SNN conversion of the original model.",
    )


if __name__ == "__main__":
    main()
