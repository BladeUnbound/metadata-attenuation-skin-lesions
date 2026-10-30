# novelty/

Maths and code for the revised CV EL paper. See [DERIVATION.md](DERIVATION.md) for the derivations; every step is labelled as proved and checked, already known, or the point where it stops.

| File | What it does |
|---|---|
| `snn_math.py` | Spiking-neuron forward pass, the hand-derived backward pass, and the closed forms from DERIVATION.md Part A |
| `test_snn_math.py` | Checks each Part A step against PyTorch autograd or exact simulation (18 tests) |
| `added_value.py` | Tests for "genetics adds value beyond MRI": likelihood-ratio test, DeLong, interaction test, Holm correction |
| `test_added_value.py` | Checks those tests' false-positive rates and power by simulation (8 tests) |

## Run

```sh
.venv/bin/python -m pytest -q
```

The `.venv` folder holds numpy, scipy, torch, scikit-learn, statsmodels and pytest. To rebuild it:

```sh
python3 -m venv .venv && .venv/bin/pip install numpy scipy torch scikit-learn statsmodels pytest
```

## Using `added_value.py` on real data

```python
from added_value import lrt_added_predictors, delong_paired, interaction_test, holm

# y: progressed within 3 years (0/1); mri: out-of-fold or external-cohort model scores
# genetics: columns for APOE e4 count and polygenic risk score
r = lrt_added_predictors(y, mri, genetics)                     # primary test
if r["p"] < 0.05:
    gain = delong_paired(y, r["pred_full"], r["pred_base"])    # report gain["diff"] and gain["ci"] only
```
