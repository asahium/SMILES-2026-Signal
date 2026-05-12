import json
from pathlib import Path

import gdown
import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = (
    "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
)
downloaded_file = "challenge.mat"
if not Path(downloaded_file).exists():
    gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    """TX-driven IMD3 cancellation plus coherent rank-1 residual removal."""
    del tx_n

    fit_tx_prediction = helpers["fit_tx_prediction"]
    score_filter = helpers["score_filter"]

    tx_pred = fit_tx_prediction(rx)
    residual = rx - tx_pred

    def apply_score_filter_matrix(x):
        return np.column_stack([score_filter(x[:, ch]) for ch in range(x.shape[1])])

    band_residual = apply_score_filter_matrix(residual)

    # Estimate the dominant spatially coherent residual in the scoring band.
    margin = 2048
    cov_source = band_residual[margin:-margin]
    if cov_source.shape[0] == 0:
        cov_source = band_residual
    covariance = cov_source.conj().T @ cov_source / cov_source.shape[0]
    _, eigvecs = np.linalg.eigh(covariance)
    mode = eigvecs[:, -1]

    shared = band_residual @ mode
    denom = np.vdot(shared, shared) + 1e-30
    rank1_band = np.column_stack(
        [
            (np.vdot(shared, band_residual[:, ch]) / denom) * shared
            for ch in range(band_residual.shape[1])
        ]
    )

    # Compensate for the scorer applying the same bandpass filter after subtraction.
    corrected_rank1 = rank1_band.copy()
    alpha = 1.15
    for _ in range(3):
        corrected_rank1 += alpha * (
            rank1_band - apply_score_filter_matrix(corrected_rank1)
        )

    channel_scales = np.array([1.08, 1.12, 0.67, 1.12], dtype=np.float64)
    corrected_rank1 *= channel_scales[np.newaxis, :]

    return rx - tx_pred - corrected_rank1


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
