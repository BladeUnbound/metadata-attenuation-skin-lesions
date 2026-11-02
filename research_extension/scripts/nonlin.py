"""Nonlinear, attribute-free transformations Z = T_phi(S) (used at inference without A).

T_phi is a residual MLP initialised at the identity: z = s + W2 relu(W1 s + b1) + b2 with W2, b2 = 0.

Objective, minimised over phi, with the adversary g_psi and the diagnosis head h_theta each minimising their own
loss in alternation:
    L_T = - lam_A * min(MSE(g(z), a), cap) + lam_D * ||z - s||^2 / d + lam_Y * CE(h(.), y)
where h sees z ("image") or [z, a] ("fused") and lam_Y = 0 gives a pure adversarial eraser.

The adversary minimises MSE(g(z), a) (or CE for categorical a). The cap (the trivial loss, Var(a) = 1 after
standardisation) stops T from being rewarded for pushing the adversary beyond chance. Both g and h are 2-layer
MLPs (hidden 128). A is standardised; continuous attributes use MSE.
"""
import numpy as np
import torch
import torch.nn as nn


class ResMLP(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)

    def forward(self, s):
        return s + self.l2(torch.relu(self.l1(s)))


def mlp(din, dout, hidden=128):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dout))


def fit_transform(S, A, y, lam_A, lam_D=1.0, lam_Y=0.0, task="image", epochs=60, batch=256, lr=1e-3, adv_steps=3, seed=0,
                  adversary="mse"):
    """Returns a callable mapping (n, d) numpy arrays to transformed arrays, plus the training history."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)
    mu, sd = S.mean(0), S.std(0) + 1e-8
    am, asd = A.mean(), A.std() + 1e-8
    Xs = torch.tensor((S - mu) / sd, dtype=torch.float32)
    As = torch.tensor((A - am) / asd, dtype=torch.float32)[:, None]
    Ys = torch.tensor(y, dtype=torch.float32)[:, None]
    d = S.shape[1]
    T = ResMLP(d)
    if adversary == "decile":  # SYN-3c: distributional adversary over training-fit age deciles
        edges = np.quantile(A, np.linspace(0, 1, 11)[1:-1])
        Ab = torch.tensor(np.digitize(A, edges), dtype=torch.long)
        g = mlp(d, 10)
    else:
        g = mlp(d, 1)
    ce = nn.CrossEntropyLoss()
    cap = float(np.log(10)) if adversary == "decile" else 1.0
    h = mlp(d + (1 if task == "fused" else 0), 1, hidden=64)
    oT = torch.optim.Adam(T.parameters(), lr=lr)
    og = torch.optim.Adam(g.parameters(), lr=lr)
    oh = torch.optim.Adam(h.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xs)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, ab, yb = Xs[idx], As[idx], Ys[idx]
            advloss = (lambda out: ce(out, Ab[idx])) if adversary == "decile" else (lambda out: ((out - ab) ** 2).mean())
            with torch.no_grad():
                zb = T(xb)
            for _ in range(adv_steps):                       # adversary catches up on the current transform
                og.zero_grad(); advloss(g(zb)).backward(); og.step()
            if lam_Y > 0:
                hin = torch.cat([zb, ab], 1) if task == "fused" else zb
                oh.zero_grad(); bce(h(hin), yb).backward(); oh.step()
            oT.zero_grad()
            z = T(xb)
            adv = torch.clamp(advloss(g(z)), max=cap)
            loss = -lam_A * adv + lam_D * ((z - xb) ** 2).sum(1).mean() / d
            if lam_Y > 0:
                hin = torch.cat([z, ab], 1) if task == "fused" else z
                loss = loss + lam_Y * bce(h(hin), yb)
            loss.backward(); oT.step()
        hist.append(float(adv))

    def transform(Snew):
        with torch.no_grad():
            z = T(torch.tensor((Snew - mu) / sd, dtype=torch.float32)).numpy()
        return z * sd + mu                                    # back to the original scale
    return transform, hist
