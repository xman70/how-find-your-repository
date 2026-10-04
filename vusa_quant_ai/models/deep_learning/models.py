"""Deep-learning forecasters (PyTorch): LSTM, GRU, Temporal CNN, Transformer encoder and
an N-BEATS-style univariate model.

* CUDA is used automatically when available; otherwise CPU (no GPU required).
* If PyTorch is not installed the models report ``available=False`` and are skipped -
  they are never replaced by a differently-named model.
* Multi-task head: expected forward return (Huber loss) + direction logit (BCE).
* Early stopping uses the *last* 15 % of the training window (time-ordered), never data
  after the training window.
* A Temporal Fusion Transformer is not included: on a single daily series with a few
  thousand observations it is not computationally/statistically practical; the
  Transformer encoder covers the attention-based family.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ..base import BaseForecaster, ModelInfo, Preprocessor

try:  # a half-installed torch or missing DLLs (WinError 126/1114) must never crash start-up
    import torch
    from torch import nn

    TORCH, _REASON = True, ""
except Exception as _exc:  # noqa: BLE001
    TORCH = False
    _REASON = (f"PyTorch could not be loaded ({type(_exc).__name__}: {_exc})"[:300]
               if importlib.util.find_spec("torch") else "PyTorch not installed - deep learning skipped")


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if TORCH:
    class _Head(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ret = nn.Linear(d, 1)
            self.cls = nn.Linear(d, 1)

        def forward(self, z):
            return self.ret(z).squeeze(-1), self.cls(z).squeeze(-1)

    class _RNN(nn.Module):
        def __init__(self, n_in, hidden=32, cell="lstm", dropout=0.2):
            super().__init__()
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn(n_in, hidden, batch_first=True, num_layers=1)
            self.drop = nn.Dropout(dropout)
            self.head = _Head(hidden)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1]))

    class _TCN(nn.Module):
        def __init__(self, n_in, ch=32, k=3, levels=4, dropout=0.2):
            super().__init__()
            layers, c_in = [], n_in
            for i in range(levels):
                d = 2 ** i
                layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(c_in, ch, k, dilation=d), nn.ReLU(),
                           nn.Dropout(dropout)]
                c_in = ch
            self.net = nn.Sequential(*layers)
            self.head = _Head(ch)

        def forward(self, x):  # causal convolutions (left padding only)
            z = self.net(x.transpose(1, 2))
            return self.head(z[:, :, -1])

    class _Transformer(nn.Module):
        def __init__(self, n_in, d=32, heads=4, layers=2, dropout=0.2, max_len=512):
            super().__init__()
            self.inp = nn.Linear(n_in, d)
            self.pos = nn.Parameter(torch.zeros(1, max_len, d))
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = _Head(d)

        def forward(self, x):
            z = self.inp(x) + self.pos[:, -x.shape[1]:]
            L = x.shape[1]
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)  # causal
            z = self.enc(z, mask=mask)
            return self.head(z[:, -1])

    class _NBeats(nn.Module):
        def __init__(self, n_in, width=64, blocks=3):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width),
                                                       nn.ReLU(), nn.Linear(width, n_in + width)) for _ in range(blocks)])
            self.head = _Head(width)
            self.n_in, self.width = n_in, width

        def forward(self, x):  # x: (B, L, 1) -> flatten
            res = x.squeeze(-1)
            agg = 0
            for b in self.blocks:
                o = b(res)
                backcast, fc = o[:, : self.n_in], o[:, self.n_in:]
                res = res - backcast
                agg = agg + fc
            return self.head(agg)


class SequenceForecaster(BaseForecaster):
    needs_context = True
    arch = "lstm"
    univariate = False

    def __init__(self, horizon, seed=42, epochs=30, lookback=40, max_features=24, lr=1e-3, batch=128, **hp):
        super().__init__(horizon, seed, **hp)
        self.epochs, self.lookback, self.max_features, self.lr, self.batch = epochs, lookback, max_features, lr, batch

    def _net(self, n_in):
        return {"lstm": lambda: _RNN(n_in, cell="lstm"), "gru": lambda: _RNN(n_in, cell="gru"),
                "tcn": lambda: _TCN(n_in), "transformer": lambda: _Transformer(n_in),
                "nbeats": lambda: _NBeats(self.lookback)}[self.arch]()

    def _matrix(self, X_full: pd.DataFrame) -> np.ndarray:
        if self.univariate:
            r = np.log(self.context_close).diff().reindex(X_full.index).fillna(0).values * 50
            return r.reshape(-1, 1).astype(np.float32)
        return self.pre_.transform(X_full[self.cols_])

    def _windows(self, M: np.ndarray, pos: np.ndarray) -> np.ndarray:
        L = self.lookback
        idx = pos[:, None] + np.arange(-L + 1, 1)[None, :]
        idx = np.clip(idx, 0, None)
        return M[idx]

    def fit(self, X, y_ret, y_up, w=None):
        if not TORCH:
            raise RuntimeError(_REASON)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Xf = self.context_X if self.context_X is not None else X
        hist = Xf.loc[: X.index[-1]]
        if not self.univariate:
            var = X.var().sort_values(ascending=False)
            self.cols_ = list(var.index[: self.max_features])
            self.pre_ = Preprocessor().fit(X[self.cols_])
        M = self._matrix(hist)
        pos = hist.index.get_indexer(X.index)
        m = (pos >= self.lookback) & y_ret.notna().values & y_up.notna().values
        pos, yr, yc = pos[m], y_ret.values[m].astype(np.float32), y_up.values[m].astype(np.float32)
        self.y_scale_ = float(np.std(yr) or 1.0)
        W = self._windows(M, pos)
        n_val = max(50, int(0.15 * len(W)))
        tr, va = slice(0, len(W) - n_val), slice(len(W) - n_val, len(W))
        dev = device()
        self.net_ = self._net(1 if self.univariate else len(self.cols_)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-3)
        Xt = torch.tensor(W, device=dev)
        Yr = torch.tensor(yr / self.y_scale_, device=dev)
        Yc = torch.tensor(yc, device=dev)
        huber, bce = nn.HuberLoss(), nn.BCEWithLogitsLoss()
        best, best_state, patience = np.inf, None, 0
        n_tr = len(W) - n_val
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n_tr, device=dev)
            for i in range(0, n_tr, self.batch):
                b = perm[i: i + self.batch]
                pr, pc = self.net_(Xt[b])
                loss = huber(pr, Yr[b]) + bce(pc, Yc[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                pr, pc = self.net_(Xt[va])
                vl = float(huber(pr, Yr[va]) + bce(pc, Yc[va]))
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        if best_state:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        with torch.no_grad():
            pr, _ = self.net_(Xt[tr])
        self.resid_std_ = float(np.std(pr.cpu().numpy() * self.y_scale_ - yr[tr]))
        self.epochs_run_ = ep + 1
        return self

    def predict(self, X):
        Xf = self.context_X if self.context_X is not None else X
        hist = Xf.loc[: X.index[-1]]
        M = self._matrix(hist)
        pos = hist.index.get_indexer(X.index)
        W = self._windows(M, pos)
        dev = device()
        out_r, out_p = [], []
        with torch.no_grad():
            for i in range(0, len(W), 1024):
                pr, pc = self.net_(torch.tensor(W[i: i + 1024], device=dev))
                out_r.append(pr.cpu().numpy() * self.y_scale_)
                out_p.append(torch.sigmoid(pc).cpu().numpy())
        return pd.DataFrame({"ret": np.concatenate(out_r), "p_up": np.clip(np.concatenate(out_p), 0.001, 0.999)},
                            index=X.index)


class LSTMForecaster(SequenceForecaster):
    arch = "lstm"
    info = ModelInfo("lstm", "deep_learning", "LSTM (1x32) multi-task sequence model", available=TORCH, unavailable_reason=_REASON)


class GRUForecaster(SequenceForecaster):
    arch = "gru"
    info = ModelInfo("gru", "deep_learning", "GRU (1x32) multi-task sequence model", available=TORCH, unavailable_reason=_REASON)


class TCNForecaster(SequenceForecaster):
    arch = "tcn"
    info = ModelInfo("tcn", "deep_learning", "Temporal CNN with causal dilated convolutions", available=TORCH,
                     unavailable_reason=_REASON)


class TransformerForecaster(SequenceForecaster):
    arch = "transformer"
    info = ModelInfo("transformer", "deep_learning", "Causal Transformer encoder (2 layers, d=32)", available=TORCH,
                     unavailable_reason=_REASON)


class NBeatsForecaster(SequenceForecaster):
    arch = "nbeats"
    univariate = True
    info = ModelInfo("nbeats", "deep_learning", "N-BEATS-style residual MLP on past returns (univariate)",
                     available=TORCH, unavailable_reason=_REASON)

    def __init__(self, horizon, seed=42, epochs=30, lookback=60, **hp):
        super().__init__(horizon, seed, epochs=epochs, lookback=lookback, **hp)


DL_MODELS = {c.info.name: c for c in (LSTMForecaster, GRUForecaster, TCNForecaster, TransformerForecaster, NBeatsForecaster)}
